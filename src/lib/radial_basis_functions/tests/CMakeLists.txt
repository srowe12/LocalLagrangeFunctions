set(test_name rbf_test)
set(test_sources rbf_test.cpp)

add_executable(${test_name} ${test_sources})
target_link_libraries(${test_name} test_main ${math_lib} ${ARMADILLO_LIBRARIES})

add_test(NAME ${test_name} COMMAND ${test_name})
set_target_properties( ${test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/testbin )

add_executable(rbf_perf_test rbf_perf_test.cpp)
target_link_libraries(rbf_perf_test ${math_lib} ${ARMADILLO_LIBRARIES})
