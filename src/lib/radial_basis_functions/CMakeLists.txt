set(rbf_lib rbf)
set(rbf_hdr rbf.h thin_plate_spline.h gaussian.h) 

add_library(${rbf_lib} interpolate.cpp gaussian.cpp thin_plate_spline.cpp)
FIND_PACKAGE(Boost)
FIND_PACKAGE(Armadillo)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR} ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(${rbf_lib} ${ARMADILLO_LIBRARIES} ${LAPACK_LIBRARIES})
ADD_DEFINITIONS("-DHAS_BOOST")
add_subdirectory(tests)


