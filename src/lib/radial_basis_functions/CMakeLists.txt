add_library(rbf interpolate.cpp gaussian.cpp thin_plate_spline.cpp)
target_link_libraries(rbf ${ARMADILLO_LIBRARIES} ${LAPACK_LIBRARIES})
add_subdirectory(tests)


