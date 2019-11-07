# https://github.com/benjaminjack/python_cpp_example

cmake_minimum_required(VERSION 2.8.12)
project(treeck)

SET(SOURCE_DIR "src/cpp")
# Tell cmake that headers are in alse in source_dir
include_directories(${SOURCE_DIR})
SET(SOURCES
    "${SOURCE_DIR}/bindings.cpp"
    "${SOURCE_DIR}/domain.cpp"
    "${SOURCE_DIR}/tree.cpp"
    )

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

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic)
endif()
