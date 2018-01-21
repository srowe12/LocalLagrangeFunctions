#ifndef SPEED_TEST_H
#define SPEED_TEST_H

#include "../local_lagrange.h"
#include "math_tools.h"
#include <gtest/gtest.h>
#include <stdio.h>
#include <string>
#include <utility>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/index/rtree.hpp>

#include <chrono>

void BuildLocalLagrange(LocalLagrangeAssembler &llc, size_t iter);

#endif // SPEED_TEST_H
