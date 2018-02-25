set(lagrange_test_src test.cpp )

add_executable(lagrange_test ${lagrange_test_src})
include_directories( ${CMAKE_LIB_DIR}/math_utils/ ${math_hdr} ${ARMADILLO_INCLUDE_DIRS} ${Boost_INCLUDE_DIR})
target_link_libraries(lagrange_test ${math_lib} ${lagrange_lib} ${ARMADILLO_LIBRARIES} ${GTEST_BOTH_LIBRARIES})

add_test(NAME lagrange_test COMMAND lagrange_test)

set(lagrange_interpolant_test_src lagrange_interpolant_test.cpp )

add_executable(lagrange_interpolant_test ${lagrange_interpolant_test_src})
include_directories( ${CMAKE_LIB_DIR}/math_utils/ ${math_hdr} ${ARMADILLO_INCLUDE_DIRS} ${Boost_INCLUDE_DIR})
target_link_libraries(lagrange_interpolant_test ${math_lib} ${lagrange_lib} ${ARMADILLO_LIBRARIES} ${GTEST_BOTH_LIBRARIES})

add_test(NAME lagrange_interpolant_test COMMAND lagrange_interpolant_test)

set(lagrange_int_test_src integration_test.cpp)
add_executable(lagrange_int_test ${lagrange_int_test_src})
include_directories( ${CMAKE_LIB_DIR}/math_utils/ ${math_hdr} ${ARMADILLO_INCLUDE_DIRS} ${Boost_INCLUDE_DIR})
target_link_libraries(lagrange_int_test ${math_lib} ${lagrange_lib} ${ARMADILLO_LIBRARIES} ${GTEST_BOTH_LIBRARIES})

add_test(NAME lagrange_int_test COMMAND lagrange_int_test)

set(lagrange_speed_test_src speed_test.cpp)
add_executable(lagrange_speed_test ${lagrange_speed_test_src})
include_directories( ${CMAKE_LIB_DIR}/math_utils/ ${math_hdr} ${ARMADILLO_INCLUDE_DIRS} ${Boost_INCLUDE_DIR})
target_link_libraries(lagrange_speed_test ${math_lib} ${lagrange_lib} ${ARMADILLO_LIBRARIES} ${GTEST_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

#set(lagrange_template_test_src test_template.cpp )

#add_executable(lagrange_template_test ${lagrange_template_test_src})
#include_directories( ${CMAKE_LIB_DIR}/math_utils/ ${math_hdr} ${ARMADILLO_INCLUDE_DIRS} ${Boost_INCLUDE_DIR})
#target_link_libraries(lagrange_template_test ${math_lib} ${lagrange_lib} ${ARMADILLO_LIBRARIES} ${GTEST_BOTH_LIBRARIES})

#add_test(NAME llf_test COMMAND lagrange_template_test)

set(lagrange_template_speed_test_src speed_test.cpp)
set(lagrange_template_speed_test_hdr speed_test.h)
add_executable(lagrange_template_speed_test ${lagrange_template_speed_test_src} ${lagrange_template_speed_test_hdr})
include_directories( ${CMAKE_LIB_DIR}/math_utils/ ${math_hdr} ${ARMADILLO_INCLUDE_DIRS} ${Boost_INCLUDE_DIR})
target_link_libraries(lagrange_template_speed_test ${math_lib} ${lagrange_lib} ${ARMADILLO_LIBRARIES} ${GTEST_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
