all: test.cpp
	g++ test.cpp -Wall -o test_executable.out -pthread -I /home/srowe/gtest-1.7.0/include -I ~/boost_1_59_0/  ../local_lagrange.o /home/srowe/gtest-1.7.0/mybuild/libgtest.a -larmadillo -std=c++11

clean:
	rm *.out
