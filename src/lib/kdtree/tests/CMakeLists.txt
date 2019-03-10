message("The CXX flags used are ${CMAKE_CXX_FLAGS}")
set(kdtree_test_src kdtree_test.cpp )

add_executable(kdtree_test ${kdtree_test_src})
include_directories( ${CMAKE_LIB_DIR}/math_utils/ ${math_hdr} ${ARMADILLO_INCLUDE_DIRS} ${Boost_INCLUDE_DIR})
target_link_libraries(kdtree_test ${math_lib} ${ARMADILLO_LIBRARIES})

add_test(NAME kdtree_test COMMAND kdtree_test)

