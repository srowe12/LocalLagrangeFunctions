set(rbf_test_src rbf_test.cpp )

add_executable(rbf_test ${rbf_test_src})
include_directories( ${CMAKE_LIB_DIR}/math_utils/ ${math_hdr} ${ARMADILLO_INCLUDE_DIRS} ${Boost_INCLUDE_DIR})
target_link_libraries(rbf_test ${math_lib} ${rbf_lib} ${ARMADILLO_LIBRARIES})

add_test(NAME rbf_test COMMAND rbf_test)

