cmake_minimum_required (VERSION 2.6)
project (locallagrange)
enable_testing()
find_package(GTest REQUIRED)
#May need to export GTEST_ROOT=googletest/build
#and put the include directory in the build also. This is a hack.
#For Armadillo, you may need to export CMAKE_PREFIX_PATH=$YOUR_ARMADILLO_ROOT
find_package(Threads)
find_package(OpenMP)
find_package(LAPACK)


# Set a default build type if none was specified
set(default_build_type "RelWithDebInfo")
 
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror -Wno-unused")
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_subdirectory(src)

