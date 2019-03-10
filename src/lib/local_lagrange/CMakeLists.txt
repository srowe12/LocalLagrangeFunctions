set(lagrange_lib lagrange)
set(lagrange_hdr local_lagrange.h local_lagrange_assembler.h local_lagrange_interpolant.h)
add_library(${lagrange_lib} local_lagrange.cpp local_lagrange_assembler.cpp local_lagrange_interpolant.cpp)

INCLUDE_DIRECTORIES( ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries( lagrange ${ARMADILLO_LIBRARIES} ${LAPACK_LIBRARIES})
subdirs(test)
