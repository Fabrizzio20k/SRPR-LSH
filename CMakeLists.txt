cmake_minimum_required(VERSION 3.31)
project(SRPR_LSH)

set(CMAKE_CXX_STANDARD 20)

add_executable(SRPR_LSH main.cpp
        src/lsh.h
        #src/plane.h
        #src/vec.h
)
