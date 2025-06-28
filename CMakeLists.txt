cmake_minimum_required(VERSION 3.30.5)
project(SRPR_LSH)

set(CMAKE_CXX_STANDARD 20)

add_executable(SRPR_LSH main.cpp
        src/lsh.h
        #src/plane.h
        #src/vec.h
)
add_executable(SRPR_LSH2 main2.cpp src/lsh.h src/plane.h src/vec.h src/DataManager.h src/MatrixFactorization.h src/Triplet.h src/SRPRModel.h
                src/MetricsCalculator.h
)