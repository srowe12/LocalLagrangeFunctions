set(rbf_test_src rbf_test.cpp )

add_executable(rbf_test ${rbf_test_src})
target_link_libraries(rbf_test test_main ${math_lib} ${rbf_lib} ${ARMADILLO_LIBRARIES})

add_test(NAME rbf_test COMMAND rbf_test)

