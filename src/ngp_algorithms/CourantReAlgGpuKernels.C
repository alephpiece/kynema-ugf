// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "CourantReAlgGpuKernels.h"

#include <Kokkos_Core.hpp>

#define KYNEMA_UGF_COURANT_RE_INLINE KOKKOS_INLINE_FUNCTION
#include "ngp_algorithms/CourantReKernelMath.h"
#undef KYNEMA_UGF_COURANT_RE_INLINE

#include "AlgTraits.h"
#include "BuildTemplates.h"
#include "Realm.h"
#include "ngp_utils/NgpFieldManager.h"
#include "ngp_utils/NgpTypes.h"

#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/Types.hpp>

#include <string>

namespace sierra {
namespace kynema_ugf {

namespace {

template <typename AlgTraits, typename NgpMeshType, typename FieldType>
KOKKOS_INLINE_FUNCTION void
load_elem_fields(
  const NgpMeshType& ngpMesh,
  const typename NgpMeshType::MeshIndex& elemIndex,
  const FieldType& coordinatesField,
  const FieldType& velocityField,
  const FieldType& densityField,
  const FieldType& viscosityField,
  double coordinates[AlgTraits::nodesPerElement_][AlgTraits::nDim_],
  double velocity[AlgTraits::nodesPerElement_][AlgTraits::nDim_],
  double density[AlgTraits::nodesPerElement_],
  double viscosity[AlgTraits::nodesPerElement_])
{
  const auto nodes = ngpMesh.get_nodes(stk::topology::ELEM_RANK, elemIndex);
  for (int n = 0; n < AlgTraits::nodesPerElement_; ++n) {
    const auto nodeIndex = ngpMesh.fast_mesh_index(nodes[n]);
    for (int d = 0; d < AlgTraits::nDim_; ++d) {
      coordinates[n][d] = coordinatesField.get(nodeIndex, d);
      velocity[n][d] = velocityField.get(nodeIndex, d);
    }
    density[n] = densityField.get(nodeIndex, 0);
    viscosity[n] = viscosityField.get(nodeIndex, 0);
  }
}

} // namespace

template <typename AlgTraits, typename MeshInfoType>
CflRe
CourantReAlgGpuKernelLauncher<AlgTraits, MeshInfoType>::execute(
  const MeshInfoType& meshInfo,
  const stk::mesh::Selector& selector,
  const unsigned coordinates,
  const unsigned velocity,
  const unsigned density,
  const unsigned viscosity,
  const unsigned elemCFL,
  const unsigned elemRe,
  const double dt)
{
  using NgpMeshType = stk::mesh::NgpMesh;
  using Traits = kynema_ugf_ngp::NGPMeshTraits<NgpMeshType>;
  using TeamPolicy = typename Traits::TeamPolicy;
  using TeamHandleType = typename Traits::TeamHandleType;
  using MeshIndex = typename Traits::MeshIndex;

  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  auto coordinatesField = fieldMgr.template get_field<double>(coordinates);
  auto velocityField = fieldMgr.template get_field<double>(velocity);
  auto densityField = fieldMgr.template get_field<double>(density);
  auto viscosityField = fieldMgr.template get_field<double>(viscosity);
  auto elemCFLField = fieldMgr.template get_field<double>(elemCFL);
  auto elemReField = fieldMgr.template get_field<double>(elemRe);

  elemCFLField.clear_sync_state();
  elemReField.clear_sync_state();
  coordinatesField.sync_to_device();
  velocityField.sync_to_device();
  densityField.sync_to_device();
  viscosityField.sync_to_device();

  const auto buckets =
    ngpMesh.get_bucket_ids(stk::topology::ELEM_RANK, selector);
  const auto teamExec = TeamPolicy(buckets.size(), NTHREADS_PER_DEVICE_TEAM);

  double cflMax = -1.0e6;
  Kokkos::Max<double> cflReducer(cflMax);
  const std::string cflAlgName =
    "CourantReAlgDirect_CFL_" + std::to_string(AlgTraits::topo_);
  Kokkos::parallel_reduce(
    cflAlgName, teamExec,
    KOKKOS_LAMBDA(const TeamHandleType& team, double& teamMax) {
      const auto bktId = buckets.device_get(team.league_rank());
      const auto& bkt = ngpMesh.get_bucket(stk::topology::ELEM_RANK, bktId);

      double bucketMax = -1.0e6;
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, bkt.size()),
        [&](const size_t& bktIndex, double& threadMax) {
          const MeshIndex elemIndex{
            bkt.bucket_id(), static_cast<unsigned>(bktIndex)};
          double elemCoordinates[AlgTraits::nodesPerElement_][AlgTraits::nDim_];
          double elemVelocity[AlgTraits::nodesPerElement_][AlgTraits::nDim_];
          double elemDensity[AlgTraits::nodesPerElement_];
          double elemViscosity[AlgTraits::nodesPerElement_];
          load_elem_fields<AlgTraits>(
            ngpMesh, elemIndex, coordinatesField, velocityField, densityField,
            viscosityField, elemCoordinates, elemVelocity, elemDensity,
            elemViscosity);
          const double elemValue = compute_elem_courant_from_arrays<AlgTraits>(
            elemCoordinates, elemVelocity, dt);
          elemCFLField.get(elemIndex, 0) = elemValue;
          threadMax = courant_re_max(threadMax, elemValue);
        },
        Kokkos::Max<double>(bucketMax));

      Kokkos::single(Kokkos::PerTeam(team), [&]() {
        teamMax = courant_re_max(teamMax, bucketMax);
      });
    },
    cflReducer);

  double reMax = -1.0e6;
  Kokkos::Max<double> reReducer(reMax);
  const std::string reAlgName =
    "CourantReAlgDirect_RE_" + std::to_string(AlgTraits::topo_);
  Kokkos::parallel_reduce(
    reAlgName, teamExec,
    KOKKOS_LAMBDA(const TeamHandleType& team, double& teamMax) {
      const auto bktId = buckets.device_get(team.league_rank());
      const auto& bkt = ngpMesh.get_bucket(stk::topology::ELEM_RANK, bktId);

      double bucketMax = -1.0e6;
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, bkt.size()),
        [&](const size_t& bktIndex, double& threadMax) {
          const MeshIndex elemIndex{
            bkt.bucket_id(), static_cast<unsigned>(bktIndex)};
          double elemCoordinates[AlgTraits::nodesPerElement_][AlgTraits::nDim_];
          double elemVelocity[AlgTraits::nodesPerElement_][AlgTraits::nDim_];
          double elemDensity[AlgTraits::nodesPerElement_];
          double elemViscosity[AlgTraits::nodesPerElement_];
          load_elem_fields<AlgTraits>(
            ngpMesh, elemIndex, coordinatesField, velocityField, densityField,
            viscosityField, elemCoordinates, elemVelocity, elemDensity,
            elemViscosity);
          const double elemValue = compute_elem_reynolds_from_arrays<AlgTraits>(
            elemCoordinates, elemVelocity, elemDensity, elemViscosity);
          elemReField.get(elemIndex, 0) = elemValue;
          threadMax = courant_re_max(threadMax, elemValue);
        },
        Kokkos::Max<double>(bucketMax));

      Kokkos::single(Kokkos::PerTeam(team), [&]() {
        teamMax = courant_re_max(teamMax, bucketMax);
      });
    },
    reReducer);

  elemCFLField.modify_on_device();
  elemReField.modify_on_device();

  CflRe result;
  result.max_cfl = cflMax;
  result.max_re = reMax;
  return result;
}

template class CourantReAlgGpuKernelLauncher<AlgTraitsHex8, Realm::NgpMeshInfo>;
template class CourantReAlgGpuKernelLauncher<AlgTraitsTet4, Realm::NgpMeshInfo>;
template class CourantReAlgGpuKernelLauncher<AlgTraitsPyr5, Realm::NgpMeshInfo>;
template class CourantReAlgGpuKernelLauncher<AlgTraitsWed6, Realm::NgpMeshInfo>;
template class CourantReAlgGpuKernelLauncher<
  AlgTraitsQuad4_2D,
  Realm::NgpMeshInfo>;
template class CourantReAlgGpuKernelLauncher<
  AlgTraitsTri3_2D,
  Realm::NgpMeshInfo>;

} // namespace kynema_ugf
} // namespace sierra
