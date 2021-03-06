CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(cudaBenchmark)

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
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDA_INCLUDE_DIRS}
    )

FILE(GLOB SOURCES
    "src/*.cu"
)

FIND_LIBRARY(CUDA_nvToolsExt_LIBRARY
	NAMES "nvToolsExt" "nvToolsExt64_1"
	PATH_SUFFIXES "lib64" "lib" "lib/x64" "lib/Win32"
	PATHS ${CUDA_TOOLKIT_ROOT_DIR}
		$ENV{NVTOOLSEXT_PATH}
	DOC "CUDA NV Tools Ext Library"
	)
IF(WIN32)
	FIND_PATH(
        CUDA_nvToolsExt_INCLUDE
        NAMES "nvToolsExt.h"
        PATHS $ENV{NVTOOLSEXT_PATH}
        PATH_SUFFIXES "include"
        DOC "NV Tools Ext Include Directory"
        NO_DEFAULT_PATH
    )
	INCLUDE_DIRECTORIES(
		${CUDA_nvToolsExt_INCLUDE}
    )
ENDIF(WIN32)

LIST(APPEND CMAKE_CXX_FLAGS "-O3 -Wall")

LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)

FOREACH(FILENAME ${SOURCES})
    STRING(REPLACE ".cu" "" EXEC ${FILENAME})
    STRING(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/src/" "" EXEC ${EXEC})
    CUDA_ADD_EXECUTABLE(${EXEC} ${FILENAME})
    TARGET_LINK_LIBRARIES(${EXEC} ${CUDA_nvToolsExt_LIBRARY})
	IF(WIN32)
		ADD_CUSTOM_COMMAND(TARGET ${EXEC}
			POST_BUILD
			COMMAND copy ARGS "\"$ENV{NVTOOLSEXT_PATH}bin\\x64\\nvToolsExt64_1.dll\" \"${PROJECT_BINARY_DIR}/\$\(Configuration\)/\""
			COMMENT "Running SomeCommand")
	ENDIF(WIN32)
ENDFOREACH()
