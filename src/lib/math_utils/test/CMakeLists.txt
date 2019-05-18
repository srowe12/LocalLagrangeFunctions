set(math_test_src math_test.cpp)
add_executable(math_test ${math_test_src})
include_directories(${math_tools_hdr})
target_link_libraries(math_test ${math_tools_lib})

#Test name then test exectuables. 
add_test(NAME math_tests COMMAND math_test)


set(polynomial_test_src polynomial_test.cpp)
add_executable(polynomial_test ${polynomial_test_src})

target_link_libraries(polynomial_test ${math_tools_lib})

#Test name then test exectuables. 
add_test(NAME polynomial_test COMMAND polynomial_test)
