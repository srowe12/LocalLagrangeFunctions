set(math_test_src math_test.cpp ../math_tools.h)
add_executable(math_test ${math_test_src})
include_directories(${math_tools_hdr})
target_link_libraries(math_test ${math_tools_lib} ${GTEST_BOTH_LIBRARIES})

#Test name then test exectuables. 
add_test(math_tests math_test)
