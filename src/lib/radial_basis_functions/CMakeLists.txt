set(rbf_lib rbf)
set(rbf_hdr rbf.h thin_plate_spline.h gaussian.h) 

add_library(${rbf_lib} interpolate.cpp gaussian.cpp thin_plate_spline.cpp)
INCLUDE_DIRECTORIES( ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(${rbf_lib} ${ARMADILLO_LIBRARIES} ${LAPACK_LIBRARIES})
add_subdirectory(tests)


