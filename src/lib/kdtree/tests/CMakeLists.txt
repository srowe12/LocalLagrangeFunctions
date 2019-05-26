set(kdtree_test_src kdtree_test.cpp )

add_executable(kdtree_test ${kdtree_test_src})
target_link_libraries(kdtree_test test_main ${math_lib} ${ARMADILLO_LIBRARIES})

add_test(NAME kdtree_test COMMAND kdtree_test)

