set(test_name lagrange_test)
set(test_sources lagrange_test.cpp)

add_executable(${test_name} ${test_sources})
target_link_libraries(${test_name} test_main ${math_lib} ${ARMADILLO_LIBRARIES})

add_test(NAME ${test_name} COMMAND ${test_name})
set_target_properties( ${test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/testbin )


#================
set(test_name lagrange_interpolant_test)
set(test_sources lagrange_interpolant_test.cpp)

add_executable(${test_name} ${test_sources})
target_link_libraries(${test_name} test_main ${math_lib} ${ARMADILLO_LIBRARIES})

add_test(NAME ${test_name} COMMAND ${test_name})
set_target_properties( ${test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/testbin )



#================
set(test_name lagrange_int_test)
set(test_sources integration_test.cpp)

add_executable(${test_name} ${test_sources})
target_link_libraries(${test_name} test_main ${math_lib} ${ARMADILLO_LIBRARIES})

add_test(NAME ${test_name} COMMAND ${test_name})
set_target_properties( ${test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/testbin )

#================
set(lagrange_speed_test_src speed_test.cpp)
add_executable(lagrange_speed_test ${lagrange_speed_test_src})
target_link_libraries(lagrange_speed_test test_main ${math_lib} ${lagrange_lib} ${ARMADILLO_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})


