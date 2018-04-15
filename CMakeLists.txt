cmake_minimum_required(VERSION 3.8)
project(ML)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        Digit_Recog_ANN.cpp
        install.sh
        Makefile)

add_executable(ML ${SOURCE_FILES})