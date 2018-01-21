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

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(${GTEST_INCLUDE_DIRS})
SET(CMAKE_CXX_FLAGS "-std=c++14 -Wall -O3")
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_subdirectory(src)

