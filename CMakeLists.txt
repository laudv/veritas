# https://github.com/benjaminjack/python_cpp_example

cmake_minimum_required(VERSION 2.8.12)
project(gbcheck)

SET(SOURCE_DIR "src/cpp")
# Tell cmake that headers are in alse in source_dir
include_directories(${SOURCE_DIR})
SET(SOURCES "${SOURCE_DIR}/bindings.cpp")

#SET(TEST_DIR "tests")
#SET(TESTS ${SOURCES}
#    "${TEST_DIR}/test_main.cpp"
#    "${TEST_DIR}/test_math.cpp")

# Generate a test executable
#include_directories(lib/catch/include)
#add_executable("${PROJECT_NAME}_test" ${TESTS})

# Generate python module
#add_subdirectory(lib/pybind11)
#pybind11_add_module(python_cpp_example ${SOURCES} "${SOURCE_DIR}/bindings.cpp")

#add_library(${PROJECT_NAME} ${SOURCES})

find_package(pybind11 REQUIRED)
pybind11_add_module(${PROJECT_NAME} ${SOURCES})
