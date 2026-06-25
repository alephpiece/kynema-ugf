set(kynema_ugf_courant_re_gpu_source
  ${CMAKE_CURRENT_SOURCE_DIR}/CourantReAlgGpuKernels.C)
set(kynema_ugf_courant_re_host_source
  ${CMAKE_CURRENT_SOURCE_DIR}/CourantReAlg.C)

if(ENABLE_ROCM)
  set_source_files_properties(
    ${kynema_ugf_courant_re_host_source}
    ${kynema_ugf_courant_re_gpu_source}
    PROPERTIES LANGUAGE HIP)

  function(kynema_ugf_copy_courant_compile_usage target_name)
    foreach(prop IN ITEMS INCLUDE_DIRECTORIES COMPILE_DEFINITIONS)
      get_target_property(prop_value ${kynema_ugf_lib_name} ${prop})
      if(prop_value AND NOT prop_value MATCHES "-NOTFOUND$")
        set_property(TARGET ${target_name} APPEND PROPERTY ${prop} ${prop_value})
      endif()
    endforeach()
  endfunction()

  add_library(kynema_ugf_courant_re_host_obj OBJECT
    ${kynema_ugf_courant_re_host_source})
  add_library(kynema_ugf_courant_re_gpu_obj OBJECT
    ${kynema_ugf_courant_re_gpu_source})

  foreach(courant_target IN ITEMS
      kynema_ugf_courant_re_host_obj
      kynema_ugf_courant_re_gpu_obj)
    kynema_ugf_copy_courant_compile_usage(${courant_target})
    target_compile_definitions(${courant_target} PRIVATE
      KOKKOS_DEPENDENCE
      __HIP_PLATFORM_AMD__=1
      __HIP_PLATFORM_HCC__=1
      __HIP_ROCclr__=1)
    foreach(imported_target IN ITEMS Kokkos::kokkos Kokkos::kokkoskernels)
      if(TARGET ${imported_target})
        target_link_libraries(${courant_target} PRIVATE ${imported_target})
      endif()
    endforeach()
    set_target_properties(${courant_target} PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      HIP_SEPARABLE_COMPILATION ON)
    if(GPU_TARGETS)
      set_target_properties(${courant_target} PROPERTIES
        HIP_ARCHITECTURES "${GPU_TARGETS}")
    endif()
    target_compile_options(${courant_target} PRIVATE
      $<$<COMPILE_LANGUAGE:HIP>:-fgpu-rdc;-Wno-deprecated-register;-Wno-return-type>)
  endforeach()

  target_sources(${kynema_ugf_lib_name} PRIVATE
    $<TARGET_OBJECTS:kynema_ugf_courant_re_host_obj>
    $<TARGET_OBJECTS:kynema_ugf_courant_re_gpu_obj>)
endif()

target_sources(${kynema_ugf_lib_name} PRIVATE
  # Algorithms
  ${CMAKE_CURRENT_SOURCE_DIR}/BuoyancySourceAlg.C
  $<$<NOT:$<BOOL:${ENABLE_ROCM}>>:${kynema_ugf_courant_re_host_source}>
  ${CMAKE_CURRENT_SOURCE_DIR}/DynamicPressureOpenAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/NodalGradEdgeAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/NodalGradElemAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/NodalGradBndryElemAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/EffDiffFluxCoeffAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/EffSSTDiffFluxCoeffAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/EnthalpyEffDiffFluxCoeffAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/MdotEdgeAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/MdotInflowAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/MdotDensityAccumAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/MdotOpenCorrectorAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/MdotOpenEdgeAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/AMSAvgMdotEdgeAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/SSTAMSAveragesAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/MetricTensorElemAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/TurbViscKsgsAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/TurbViscSSTAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/TurbViscSSTLRAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/TurbViscKEAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/TurbViscKOAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/WallFuncGeometryAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/ABLWallFrictionVelAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/ABLWallFluxesAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/TKEWallFuncAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/GeometryInteriorAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/GeometryBoundaryAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/SDRLowReWallAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/SDRWallFuncAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/NodalGradPOpenBoundaryAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/SSTMaxLengthScaleAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/MomentumABLWallFuncMaskUtil.C
  # Algorithm Drivers
  ${CMAKE_CURRENT_SOURCE_DIR}/CourantReAlgDriver.C
  ${CMAKE_CURRENT_SOURCE_DIR}/FieldUpdateAlgDriver.C
  ${CMAKE_CURRENT_SOURCE_DIR}/NgpAlgDriver.C
  ${CMAKE_CURRENT_SOURCE_DIR}/MdotAlgDriver.C
  ${CMAKE_CURRENT_SOURCE_DIR}/NodalGradAlgDriver.C
  ${CMAKE_CURRENT_SOURCE_DIR}/NodalBuoyancyAlgDriver.C
  ${CMAKE_CURRENT_SOURCE_DIR}/TKEWallFuncAlgDriver.C
  ${CMAKE_CURRENT_SOURCE_DIR}/GeometryAlgDriver.C
  ${CMAKE_CURRENT_SOURCE_DIR}/WallFricVelAlgDriver.C
  ${CMAKE_CURRENT_SOURCE_DIR}/SDRWallFuncAlgDriver.C
  ${CMAKE_CURRENT_SOURCE_DIR}/SSTMaxLengthScaleDriver.C
  )
