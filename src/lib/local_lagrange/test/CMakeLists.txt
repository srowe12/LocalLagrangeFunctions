set(lagrange_test_src test.cpp)

add_executable(lagrange_test ${lagrange_test_src})
include_directories( ${CMAKE_LIB_DIR}/math_utils/ ${math_hdr} ${ARMADILLO_INCLUDE_DIRS} ${Boost_INCLUDE_DIR})
target_link_libraries(lagrange_test ${math_lib} ${lagrange_lib} ${ARMADILLO_LIBRARIES} ${GTEST_BOTH_LIBRARIES})

add_test(llf_test lagrange_test)
