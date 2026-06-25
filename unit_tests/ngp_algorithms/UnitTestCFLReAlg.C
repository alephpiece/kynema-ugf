// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <random>

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestHelperObjects.h"

#include "AlgTraits.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementRepo.h"
#include "ngp_algorithms/CourantReAlg.h"
#include "ngp_algorithms/CourantReAlgDriver.h"
#include "ngp_algorithms/CourantReKernelMath.h"
#include "utils/StkHelpers.h"

namespace {

template <typename AlgTraits>
void
expect_adjacent_nodes_match_master_element()
{
  const auto* meSCS =
    sierra::kynema_ugf::MasterElementRepo::get_volume_master_element_on_host(
      AlgTraits::topo_);
  ASSERT_NE(nullptr, meSCS);

  const int* lrscv = meSCS->adjacentNodes();
  ASSERT_NE(nullptr, lrscv);

  for (int ip = 0; ip < AlgTraits::numScsIp_; ++ip) {
    EXPECT_EQ(
      lrscv[2 * ip],
      sierra::kynema_ugf::CourantReTopology<AlgTraits>::adjacent_node(ip, 0));
    EXPECT_EQ(
      lrscv[2 * ip + 1],
      sierra::kynema_ugf::CourantReTopology<AlgTraits>::adjacent_node(ip, 1));
  }
}

} // namespace

TEST(CourantReTopology, adjacentNodesMatchMasterElements)
{
  expect_adjacent_nodes_match_master_element<
    sierra::kynema_ugf::AlgTraitsHex8>();
  expect_adjacent_nodes_match_master_element<
    sierra::kynema_ugf::AlgTraitsTet4>();
  expect_adjacent_nodes_match_master_element<
    sierra::kynema_ugf::AlgTraitsPyr5>();
  expect_adjacent_nodes_match_master_element<
    sierra::kynema_ugf::AlgTraitsWed6>();
  expect_adjacent_nodes_match_master_element<
    sierra::kynema_ugf::AlgTraitsQuad4_2D>();
  expect_adjacent_nodes_match_master_element<
    sierra::kynema_ugf::AlgTraitsTri3_2D>();
}

TEST_F(MomentumKernelHex8Mesh, NGP_courant_reynolds)
{
  auto& elemCourant =
    meta_->declare_field<double>(stk::topology::ELEM_RANK, "element_courant");
  auto& elemReynolds =
    meta_->declare_field<double>(stk::topology::ELEM_RANK, "element_reynolds");
  stk::mesh::put_field_on_mesh(elemCourant, meta_->universal_part(), nullptr);
  stk::mesh::put_field_on_mesh(elemReynolds, meta_->universal_part(), nullptr);
  fill_mesh_and_init_fields();

  std::mt19937 rng;
  rng.seed(std::mt19937::default_seed);
  std::uniform_real_distribution<double> rand_num(-1.0, 1.0);

  const double dt = 0.1;
  const double velVal = 10.0 + rand_num(rng);
  const double rhoVal = 1.0 + 0.1 * rand_num(rng);
  const double viscVal = 1.0e-5 * (1.0 + rand_num(rng));

  const double reyNum = velVal / (viscVal / rhoVal + 1.0e-16);
  const double cfl = velVal * dt;

  stk::mesh::field_fill(velVal, *velocity_);
  velocity_->modify_on_host();
  velocity_->sync_to_device();

  stk::mesh::field_fill(rhoVal, *density_);
  density_->modify_on_host();
  density_->sync_to_device();

  stk::mesh::field_fill(viscVal, *viscosity_);
  viscosity_->modify_on_host();
  viscosity_->sync_to_device();

  sierra::kynema_ugf::TimeIntegrator timeIntegrator;
  timeIntegrator.timeStepN_ = dt;
  timeIntegrator.timeStepNm1_ = dt;
  timeIntegrator.gamma1_ = 1.0;
  timeIntegrator.gamma2_ = -1.0;
  timeIntegrator.gamma3_ = 0.0;

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);
  helperObjs.realm.timeIntegrator_ = &timeIntegrator;

  sierra::kynema_ugf::CourantReAlgDriver algDriver(helperObjs.realm);
  algDriver.register_elem_algorithm<sierra::kynema_ugf::CourantReAlg>(
    sierra::kynema_ugf::INTERIOR, partVec_[0], "courant_reynolds", algDriver);

  algDriver.execute();

  EXPECT_NEAR(helperObjs.realm.maxCourant_, cfl, 1.0e-14);
  EXPECT_NEAR(helperObjs.realm.maxReynolds_, reyNum, 1.0e-14);

  const auto& fieldMgr = helperObjs.realm.mesh_info().ngp_field_manager();
  auto ngpElemCourant =
    fieldMgr.get_field<double>(elemCourant.mesh_meta_data_ordinal());
  auto ngpElemReynolds =
    fieldMgr.get_field<double>(elemReynolds.mesh_meta_data_ordinal());
  ngpElemCourant.modify_on_device();
  ngpElemCourant.sync_to_host();
  ngpElemReynolds.modify_on_device();
  ngpElemReynolds.sync_to_host();

  stk::mesh::Selector sel = meta_->universal_part();
  const auto& bkts = bulk_->get_buckets(stk::topology::ELEM_RANK, sel);
  int numElemsChecked = 0;
  for (const auto* b : bkts) {
    for (const auto elem : *b) {
      const double* elemCFL = stk::mesh::field_data(elemCourant, elem);
      const double* elemRe = stk::mesh::field_data(elemReynolds, elem);
      ASSERT_NE(nullptr, elemCFL);
      ASSERT_NE(nullptr, elemRe);
      EXPECT_NEAR(elemCFL[0], cfl, 1.0e-14);
      EXPECT_NEAR(elemRe[0], reyNum, 1.0e-14);
      ++numElemsChecked;
    }
  }
  EXPECT_EQ(numElemsChecked, 1);

  // Check changing of dt
  const auto dt_get = timeIntegrator.get_time_step();
  EXPECT_NEAR(dt_get, dt, 1.0e-14);
  timeIntegrator.set_time_step(dt + 1.);
  const auto dt_get_again = timeIntegrator.get_time_step();
  EXPECT_NEAR(dt_get_again, dt + 1.0, 1.0e-14);
}
