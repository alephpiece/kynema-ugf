// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <limits>
#include <type_traits>

#include "ngp_algorithms/CourantReAlg.h"
#include "BuildTemplates.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementRepo.h"
#include "ngp_algorithms/CourantReAlgDriver.h"
#include "CourantReAlgGpuKernels.h"

#if !defined(KOKKOS_ENABLE_HIP)
#include "ngp_algorithms/CourantReReduceHelper.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldOps.h"
#include "ngp_utils/NgpReduceUtils.h"
#include "ngp_utils/NgpFieldManager.h"
#include "ScratchViews.h"
#include <stk_mesh/base/NgpMesh.hpp>
#endif

#include "Realm.h"
#include "SolutionOptions.h"
#include "utils/StkHelpers.h"

namespace sierra {
namespace kynema_ugf {

template <typename AlgTraits>
CourantReAlg<AlgTraits>::CourantReAlg(
  Realm& realm, stk::mesh::Part* part, CourantReAlgDriver& algDriver)
  : Algorithm(realm, part),
    algDriver_(algDriver),
    elemData_(realm.meta_data()),
    coordinates_(
      get_field_ordinal(realm_.meta_data(), realm_.get_coordinates_name())),
    velocity_(get_field_ordinal(
      realm_.meta_data(),
      realm_.does_mesh_move() ? "velocity_rtm" : "velocity")),
    density_(get_field_ordinal(realm_.meta_data(), "density")),
    viscosity_(get_field_ordinal(
      realm_.meta_data(),
      realm_.is_turbulent() ? "effective_viscosity_u" : "viscosity")),
    elemCFL_(get_field_ordinal(
      realm_.meta_data(), "element_courant", stk::topology::ELEM_RANK)),
    elemRe_(get_field_ordinal(
      realm_.meta_data(), "element_reynolds", stk::topology::ELEM_RANK)),
    meSCS_(
      MasterElementRepo::get_surface_master_element_on_dev(AlgTraits::topo_))
{
  elemData_.add_cvfem_surface_me(meSCS_);

  elemData_.add_coordinates_field(
    coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
  elemData_.add_gathered_nodal_field(density_, 1);
  elemData_.add_gathered_nodal_field(velocity_, AlgTraits::nDim_);
  elemData_.add_gathered_nodal_field(viscosity_, 1);
}

template <typename AlgTraits>
void
CourantReAlg<AlgTraits>::execute()
{
  const auto& meshInfo = realm_.mesh_info();

  const stk::mesh::Selector sel = realm_.meta_data().locally_owned_part() &
                                  stk::mesh::selectUnion(partVec_) &
                                  !(realm_.get_inactive_selector());

#if defined(KOKKOS_ENABLE_HIP)
  using MeshInfoType = std::decay_t<decltype(meshInfo)>;
  const CflRe cflReMax =
    CourantReAlgGpuKernelLauncher<AlgTraits, MeshInfoType>::execute(
      meshInfo, sel, coordinates_, velocity_, density_, viscosity_, elemCFL_,
      elemRe_, realm_.get_time_step());

  algDriver_.update_max_cfl_rey(cflReMax.max_cfl, cflReMax.max_re);
#else
  using ElemSimdDataType = kynema_ugf_ngp::ElemSimdData<stk::mesh::NgpMesh>;

  const auto& ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  auto& ngpCFL = fieldMgr.template get_field<double>(elemCFL_);
  auto& ngpRe = fieldMgr.template get_field<double>(elemRe_);

  const unsigned coordID = coordinates_;
  const unsigned velID = velocity_;
  const unsigned rhoID = density_;
  const unsigned viscID = viscosity_;
  const DoubleType dt = realm_.get_time_step();
  const DoubleType small = 1.0e-16;
  MasterElement* meSCS = meSCS_;

  auto numScsIp = AlgTraits::numScsIp_;
  auto nDim = AlgTraits::nDim_;

  const auto cflOps = kynema_ugf_ngp::simd_elem_field_updater(ngpMesh, ngpCFL);
  const auto reyOps = kynema_ugf_ngp::simd_elem_field_updater(ngpMesh, ngpRe);

  CflRe cflReMax;
  CflReMax<> reducer(cflReMax);

  const std::string algName =
    "CourantReAlg_" + std::to_string(AlgTraits::topo_);
  kynema_ugf_ngp::run_elem_par_reduce(
    algName, meshInfo, stk::topology::ELEM_RANK, elemData_, sel,
    KOKKOS_LAMBDA(ElemSimdDataType & edata, CflRe & threadVal) {
      auto& scrViews = edata.simdScrView;
      const auto& v_coords = scrViews.get_scratch_view_2D(coordID);
      const auto& v_vel = scrViews.get_scratch_view_2D(velID);
      const auto& v_rho = scrViews.get_scratch_view_1D(rhoID);
      const auto& v_visc = scrViews.get_scratch_view_1D(viscID);

      DoubleType elemRe = -1.0;
      DoubleType elemCFL = -1.0;

      const int* lrscv = meSCS->adjacentNodes();
      for (int ip = 0; ip < numScsIp; ++ip) {
        const int il = lrscv[2 * ip];
        const int ir = lrscv[2 * ip + 1];

        DoubleType udotx = 0.0;
        DoubleType dxSq = 0.0;
        for (int d = 0; d < nDim; ++d) {
          DoubleType uIp = 0.5 * (v_vel(ir, d) + v_vel(il, d));
          DoubleType dxj = (v_coords(ir, d) - v_coords(il, d));
          udotx += dxj * uIp;
          dxSq += dxj * dxj;
        }

        udotx = stk::math::abs(udotx);
        const DoubleType cflIp = stk::math::abs(udotx * dt / dxSq);

        const DoubleType diffIp =
          0.5 * (v_visc(il) / v_rho(il) + v_visc(ir) / v_rho(ir)) + small;
        const DoubleType reyIp = udotx / diffIp;

        elemRe = stk::math::max(elemRe, reyIp);
        elemCFL = stk::math::max(elemCFL, cflIp);
      }
      reyOps(edata, 0) = elemRe;
      cflOps(edata, 0) = elemCFL;

      for (int i = 0; i < edata.numSimdElems; ++i) {
        threadVal.max_cfl = stk::math::max(threadVal.max_cfl, elemCFL[i]);
        threadVal.max_re = stk::math::max(threadVal.max_re, elemRe[i]);
      }
    },
    reducer);

  // Accumulate max values for all topology types
  algDriver_.update_max_cfl_rey(cflReMax.max_cfl, cflReMax.max_re);

  ngpCFL.modify_on_device();
  ngpRe.modify_on_device();
#endif
}

INSTANTIATE_KERNEL(CourantReAlg)

} // namespace kynema_ugf
} // namespace sierra
