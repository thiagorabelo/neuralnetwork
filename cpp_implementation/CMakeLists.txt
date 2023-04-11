cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(NeuralNetwork LANGUAGES CXX)

add_executable(neuralnetwork "")

target_sources(
    neuralnetwork
    PRIVATE
        main.cpp
        # src/matrix.cpp
)

list(
    APPEND
    flags
        "-Wall"
        "-Werror"
        "-Wpedantic"
)

target_include_directories(
    neuralnetwork
    PUBLIC
        "include"
)

target_compile_options(
    neuralnetwork
    PRIVATE
        ${flags}
)

set_target_properties(
    neuralnetwork
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)
