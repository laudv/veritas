# https://github.com/benjaminjack/python_cpp_example

cmake_minimum_required(VERSION 3.1)
project(veritas)

set(SOURCE_DIR "src/cpp")
# Tell cmake that headers are also in source_dir
include_directories(${SOURCE_DIR})
set(SOURCES
    "${SOURCE_DIR}/domain.cpp"
    "${SOURCE_DIR}/tree.cpp"
    "${SOURCE_DIR}/graph.cpp"
    "${SOURCE_DIR}/smt.cpp"
    )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -pedantic)
    add_compile_options(-march=native)
endif()

#
set(Z3_INSTALL "" CACHE PATH "Custom Z3 install path")
if (NOT "${Z3_INSTALL}" STREQUAL "")
    link_directories("${Z3_INSTALL}/build")
    include_directories("${Z3_INSTALL}/src/api")
    include_directories("${Z3_INSTALL}/src/api/c++")
else()
    include_directories("/usr/include/z3")
endif ()

# GProf
#cmake -DCMAKE_CXX_FLAGS=-pg -DCMAKE_EXE_LINKER_FLAGS=-pg -DCMAKE_SHARED_LINKER_FLAGS=-pg <SOURCE_DIR>


#option(USE_SHARED_LIB "Use a shared libveritas.so lib" ON)
#if (USE_SHARED_LIB)
    add_library(${PROJECT_NAME} STATIC ${SOURCES})
#endif()

# multi-threading
find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)

# systems usually have old version of cereal, so we use a git submodule
include_directories(lib/cereal/include)

option(BUILD_PYTHON_BINDINGS "Build C++ to Python bindings" ON)
if (BUILD_PYTHON_BINDINGS)
    #find_package(pybind11 REQUIRED)
    add_subdirectory(lib/pybind11)
    #pybind11_add_module("py${PROJECT_NAME}" ${SOURCES} "${SOURCE_DIR}/bindings.cpp")
    pybind11_add_module("py${PROJECT_NAME}" "${SOURCE_DIR}/bindings.cpp")
    target_link_libraries("py${PROJECT_NAME}" PRIVATE ${PROJECT_NAME})

    # Use Ubuntu's z3 (libz3-dev) install
    #find_package(Z3 REQUIRED)
    target_link_libraries("py${PROJECT_NAME}" PRIVATE -lz3)

endif (BUILD_PYTHON_BINDINGS)

option(BUILD_TESTS "Build C++ test files" OFF)
if (BUILD_TESTS)
    SET(TEST_DIR "tests")
    SET(TESTS
        "${TEST_DIR}/test_main.cpp")

    # Generate a test executable
    #include_directories(lib/catch/include)
    add_executable("test${PROJECT_NAME}" ${TESTS})
    target_link_libraries("test${PROJECT_NAME}" PRIVATE "${PROJECT_NAME}")
    target_link_libraries("test${PROJECT_NAME}" PRIVATE -lz3)
endif (BUILD_TESTS)
