// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef COURANTREALGGPUKERNELS_H
#define COURANTREALGGPUKERNELS_H

#include "ngp_algorithms/CourantReReduceHelper.h"

#include <stk_mesh/base/Selector.hpp>

namespace sierra {
namespace kynema_ugf {

template <typename AlgTraits, typename MeshInfoType>
class CourantReAlgGpuKernelLauncher
{
public:
  static CflRe execute(
    const MeshInfoType& meshInfo,
    const stk::mesh::Selector& selector,
    unsigned coordinates,
    unsigned velocity,
    unsigned density,
    unsigned viscosity,
    unsigned elemCFL,
    unsigned elemRe,
    double dt);
};

} // namespace kynema_ugf
} // namespace sierra

#endif /* COURANTREALGGPUKERNELS_H */
