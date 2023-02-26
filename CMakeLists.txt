# https://github.com/benjaminjack/python_cpp_example

cmake_minimum_required(VERSION 3.1)
project(veritas)

set(SOURCE_DIR "src/cpp")
# Tell cmake that headers are also in source_dir
include_directories(${SOURCE_DIR})
set(SOURCES
    "${SOURCE_DIR}/tree.cpp"
    "${SOURCE_DIR}/addtree.cpp"
    "${SOURCE_DIR}/json_io.cpp"
    )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    #add_compile_options(/W4 /WX)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -pedantic)# -Wconversion)
    #add_compile_options(-march=native)
endif()

# https://medium.com/@alasher/colored-c-compiler-output-with-ninja-clang-gcc-10bfe7f2b949
option (FORCE_COLORED_OUTPUT "Always produce ANSI-colored output (GNU/Clang only)." FALSE)
if (${FORCE_COLORED_OUTPUT})
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
       add_compile_options (-fdiagnostics-color=always)
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
       add_compile_options (-fcolor-diagnostics)
    endif ()
endif ()

# GProf
#cmake -DCMAKE_CXX_FLAGS=-pg -DCMAKE_EXE_LINKER_FLAGS=-pg -DCMAKE_SHARED_LINKER_FLAGS=-pg <SOURCE_DIR>


#option(USE_SHARED_LIB "Use a shared libveritas.so lib" ON)
#if (USE_SHARED_LIB)
    add_library(${PROJECT_NAME} STATIC ${SOURCES})
#endif()

# multi-threading
#find_package(Threads REQUIRED)
#target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)

option(VERITAS_SANITY_CHECKS
    "Enable additional sanity checks slowing down Veritas" OFF)
if (VERITAS_SANITY_CHECKS)
    target_compile_definitions(${PROJECT_NAME} PUBLIC VERITAS_SANITY_CHECKS=1)
else()
    target_compile_definitions(${PROJECT_NAME} PUBLIC VERITAS_SANITY_CHECKS=0)
endif (VERITAS_SANITY_CHECKS)

option(BUILD_PYTHON_BINDINGS "Build C++ to Python bindings" ON)
if (BUILD_PYTHON_BINDINGS)
    #find_package(pybind11 REQUIRED)
    add_subdirectory(lib/pybind11)
    pybind11_add_module("py${PROJECT_NAME}" "${SOURCE_DIR}/bindings.cpp")
    target_link_libraries("py${PROJECT_NAME}" PRIVATE ${PROJECT_NAME})
endif (BUILD_PYTHON_BINDINGS)

option(BUILD_TESTS "Build C++ test files" OFF)
if (BUILD_TESTS)
    include(CTest)
    set(TEST_DIR "tests")
    set(TESTS
        "${TEST_DIR}/test_main.cpp"
        "${TEST_DIR}/test_template.cpp"
        "${TEST_DIR}/test_interval.cpp"
        "${TEST_DIR}/test_box.cpp"
        "${TEST_DIR}/test_tree.cpp"
        "${TEST_DIR}/test_json_io.cpp"
        "${TEST_DIR}/test_fp.cpp"
    )

    # Generate a test executable
    add_executable("test${PROJECT_NAME}" ${TESTS})
    target_link_libraries("test${PROJECT_NAME}" "${PROJECT_NAME}")

    add_test(NAME test_template
        COMMAND "test${PROJECT_NAME}" template)
    add_test(NAME test_interval
        COMMAND "test${PROJECT_NAME}" interval)
    add_test(NAME test_box
        COMMAND "test${PROJECT_NAME}" box)
    add_test(NAME test_tree
        COMMAND "test${PROJECT_NAME}" tree)
    add_test(NAME test_json_io
        COMMAND "test${PROJECT_NAME}" json_io)
    add_test(NAME test_fp
        COMMAND "test${PROJECT_NAME}" fp)
endif (BUILD_TESTS)
