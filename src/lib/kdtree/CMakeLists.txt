
set(kdtree_src kdtree.cpp)

add_library(kdtree ${kdtree_src})
#target_link_libraries( lagrange ${ARMADILLO_LIBRARIES})



set(kdtree_test_src test/test_kdtree.cpp)
add_executable(kdtree_test ${kdtree_test_src})
target_link_libraries(kdtree_test ${kdtree} ${ARMADILLO_LIBRARIES} ${GTEST_BOTH_LIBRARIES})

#Test name then test exectuables. 
add_test(NAME kdtree_test COMMAND kdtree_test)

