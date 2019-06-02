
set(test_name lagrange_functions_test)
set(test_sources test_lagrange_functions.cpp)

add_executable(${test_name} ${test_sources})
target_link_libraries(${test_name} test_main ${math_lib} ${ARMADILLO_LIBRARIES})

add_test(NAME ${test_name} COMMAND ${test_name})
set_target_properties( ${test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/testbin )


#=================
set(test_name lagrange_functions_speed_test)
set(test_sources lagrange_functions_speed_test.cpp)

add_executable(${test_name} ${test_sources})
target_link_libraries(${test_name} ${math_lib} ${ARMADILLO_LIBRARIES})