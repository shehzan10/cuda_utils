CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(cudaDeviceProp)

# Set a default build type if none was specified
IF(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
      "MinSizeRel" "RelWithDebInfo")
ENDIF()

FIND_PACKAGE(CUDA REQUIRED)
INCLUDE_DIRECTORIES(
    ${CMAKE_INCLUDE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
    )

FILE(GLOB cuda_sources
    "*.cu"
    )

LIST(APPEND CMAKE_CXX_FLAGS "-O3 -Wall")

#LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)

CUDA_ADD_EXECUTABLE(cudaDeviceProp ${cuda_sources})
