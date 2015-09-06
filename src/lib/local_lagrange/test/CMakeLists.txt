set(lagrange_test_src test.cpp ../local_lagrange.h)
add_executable(lagrange_test ${lagrange_test_src})
include_directories(${lagrange_hdr} ${ARMADILLO_INCLUDE_DIRS} ${Boost_INCLUDE_DIR})
target_link_libraries(lagrange_test ${lagrange_lib} ${ARMADILLO_LIBRARIES} ${GTEST_BOTH_LIBRARIES})

add_test(llf_test lagrange_test)
