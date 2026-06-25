// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef COURANTREKERNELMATH_H
#define COURANTREKERNELMATH_H

#ifndef KYNEMA_UGF_COURANT_RE_INLINE
#define KYNEMA_UGF_COURANT_RE_INLINE inline
#endif

namespace sierra {
namespace kynema_ugf {

template <typename>
inline constexpr bool unsupported_courant_re_topology = false;

template <typename AlgTraits>
struct CourantReTopology
{
  static constexpr int nDim = AlgTraits::nDim_;
  static constexpr int numScsIp = AlgTraits::numScsIp_;

  KYNEMA_UGF_COURANT_RE_INLINE
  static int adjacent_node(const int ip, const int side)
  {
    if constexpr (
      AlgTraits::nDim_ == 3 && AlgTraits::nodesPerElement_ == 8 &&
      AlgTraits::numScsIp_ == 12) {
      constexpr int lrscv[24] = {0, 1, 1, 2, 2, 3, 0, 3, 4, 5, 5, 6,
                                 6, 7, 4, 7, 0, 4, 1, 5, 2, 6, 3, 7};
      return lrscv[2 * ip + side];
    } else if constexpr (
      AlgTraits::nDim_ == 3 && AlgTraits::nodesPerElement_ == 4 &&
      AlgTraits::numScsIp_ == 6) {
      constexpr int lrscv[12] = {0, 1, 1, 2, 0, 2, 0, 3, 1, 3, 2, 3};
      return lrscv[2 * ip + side];
    } else if constexpr (
      AlgTraits::nDim_ == 3 && AlgTraits::nodesPerElement_ == 5 &&
      AlgTraits::numScsIp_ == 12) {
      constexpr int lrscv[24] = {0, 1, 1, 2, 2, 3, 0, 3, 0, 4, 0, 4,
                                 1, 4, 1, 4, 2, 4, 2, 4, 3, 4, 3, 4};
      return lrscv[2 * ip + side];
    } else if constexpr (
      AlgTraits::nDim_ == 3 && AlgTraits::nodesPerElement_ == 6 &&
      AlgTraits::numScsIp_ == 9) {
      constexpr int lrscv[18] = {0, 1, 1, 2, 0, 2, 3, 4, 4,
                                 5, 3, 5, 0, 3, 1, 4, 2, 5};
      return lrscv[2 * ip + side];
    } else if constexpr (
      AlgTraits::nDim_ == 2 && AlgTraits::nodesPerElement_ == 4 &&
      AlgTraits::numScsIp_ == 4) {
      constexpr int lrscv[8] = {0, 1, 1, 2, 2, 3, 0, 3};
      return lrscv[2 * ip + side];
    } else if constexpr (
      AlgTraits::nDim_ == 2 && AlgTraits::nodesPerElement_ == 3 &&
      AlgTraits::numScsIp_ == 3) {
      constexpr int lrscv[6] = {0, 1, 1, 2, 0, 2};
      return lrscv[2 * ip + side];
    } else {
      static_assert(
        unsupported_courant_re_topology<AlgTraits>,
        "Courant/Reynolds GPU kernel does not support this topology");
      return -1;
    }
  }
};

KYNEMA_UGF_COURANT_RE_INLINE
double
courant_re_abs(const double value)
{
  return value < 0.0 ? -value : value;
}

KYNEMA_UGF_COURANT_RE_INLINE
double
courant_re_max(const double lhs, const double rhs)
{
  return lhs > rhs ? lhs : rhs;
}

template <typename AlgTraits>
KYNEMA_UGF_COURANT_RE_INLINE double
compute_elem_courant_from_arrays(
  const double coordinates[AlgTraits::nodesPerElement_][AlgTraits::nDim_],
  const double velocity[AlgTraits::nodesPerElement_][AlgTraits::nDim_],
  const double dt)
{
  using Topology = CourantReTopology<AlgTraits>;

  double elemCFL = -1.0;
  for (int ip = 0; ip < Topology::numScsIp; ++ip) {
    const int il = Topology::adjacent_node(ip, 0);
    const int ir = Topology::adjacent_node(ip, 1);

    double udotx = 0.0;
    double dxSq = 0.0;
    for (int d = 0; d < Topology::nDim; ++d) {
      const double uIp = 0.5 * (velocity[ir][d] + velocity[il][d]);
      const double dxj = coordinates[ir][d] - coordinates[il][d];
      udotx += dxj * uIp;
      dxSq += dxj * dxj;
    }

    const double cflIp = courant_re_abs(udotx * dt / dxSq);
    elemCFL = courant_re_max(elemCFL, cflIp);
  }

  return elemCFL;
}

template <typename AlgTraits>
KYNEMA_UGF_COURANT_RE_INLINE double
compute_elem_reynolds_from_arrays(
  const double coordinates[AlgTraits::nodesPerElement_][AlgTraits::nDim_],
  const double velocity[AlgTraits::nodesPerElement_][AlgTraits::nDim_],
  const double density[AlgTraits::nodesPerElement_],
  const double viscosity[AlgTraits::nodesPerElement_])
{
  using Topology = CourantReTopology<AlgTraits>;

  constexpr double small = 1.0e-16;
  double elemRe = -1.0;
  for (int ip = 0; ip < Topology::numScsIp; ++ip) {
    const int il = Topology::adjacent_node(ip, 0);
    const int ir = Topology::adjacent_node(ip, 1);

    double udotx = 0.0;
    for (int d = 0; d < Topology::nDim; ++d) {
      const double uIp = 0.5 * (velocity[ir][d] + velocity[il][d]);
      const double dxj = coordinates[ir][d] - coordinates[il][d];
      udotx += dxj * uIp;
    }

    const double diffIp =
      0.5 * (viscosity[il] / density[il] + viscosity[ir] / density[ir]) + small;
    const double reyIp = courant_re_abs(udotx) / diffIp;
    elemRe = courant_re_max(elemRe, reyIp);
  }

  return elemRe;
}

} // namespace kynema_ugf
} // namespace sierra

#endif /* COURANTREKERNELMATH_H */
