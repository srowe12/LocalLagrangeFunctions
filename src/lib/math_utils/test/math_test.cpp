#include <gtest/gtest.h>
#include "../math_tools.h"
#include <stdio.h>

TEST(MathTest,FakeTest){
   double a = 0;
   double b = 1;
   unsigned int num_points = 50;
   std::vector<double> points = mathtools::linspace<double>(a,b,num_points); 
   EXPECT_EQ(num_points+1,points.size());
   EXPECT_DOUBLE_EQ(a, points[0]);
   EXPECT_DOUBLE_EQ(b,points[points.size()-1]);

   a = 3;
   b = 4;
   num_points = 23;
   std::vector<double> points_odd =  mathtools::linspace<double>(a,b,num_points);
   EXPECT_EQ(num_points+1,points_odd.size());
   EXPECT_DOUBLE_EQ(a, points_odd[0]);
   EXPECT_DOUBLE_EQ(b, points_odd[num_points]);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  int return_value = RUN_ALL_TESTS();

  return return_value;
}
