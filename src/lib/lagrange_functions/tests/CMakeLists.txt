
add_executable(lagrange_functions_test test_lagrange_functions.cpp)
target_link_libraries(lagrange_functions_test ${math_lib} ${ARMADILLO_LIBRARIES})

add_test(NAME lagrange_functions_test COMMAND lagrange_functions_test)