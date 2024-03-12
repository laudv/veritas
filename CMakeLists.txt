# https://github.com/benjaminjack/python_cpp_example

cmake_minimum_required(VERSION 3.18)
project(veritas)

set(VERITAS_CORE, "${PROJECT_NAME}_core")

set(SOURCE_DIR_PY "src/python")
set(SOURCE_DIR "src/cpp")
set(BINDINGS_DIR "src/bindings")

# Tell cmake that headers are also in source_dir
include_directories(${SOURCE_DIR})
set(SOURCES
    "${SOURCE_DIR}/tree.cpp"
    "${SOURCE_DIR}/addtree.cpp"
    "${SOURCE_DIR}/json_io.cpp"
    "${SOURCE_DIR}/json_io_nlohmann.cpp"
    "${SOURCE_DIR}/fp_search.cpp"
    )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    #add_compile_options(/W4 /WX)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -pedantic -Wconversion)
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

# Veritas static library
add_library(${PROJECT_NAME} STATIC ${SOURCES})

# GProf
#cmake -DCMAKE_CXX_FLAGS=-pg -DCMAKE_EXE_LINKER_FLAGS=-pg -DCMAKE_SHARED_LINKER_FLAGS=-pg <SOURCE_DIR>

# nlohmann JSON
# https://github.com/nlohmann/json/tree/master#embedded

# Typically you don't care so much for a third party library's tests to be
# run from your own project's code.
set(JSON_BuildTests OFF CACHE INTERNAL "")

# If you only include this third party in PRIVATE source files, you do not
# need to install it when your main project gets installed.
set(JSON_Install OFF CACHE INTERNAL "")

# Don't use include(nlohmann_json/CMakeLists.txt) since that carries with it
# unintended consequences that will break the build.  It's generally
# discouraged (although not necessarily well documented as such) to use
# include(...) for pulling in other CMake projects anyways.
add_subdirectory(lib/nlohmann_json)

target_link_libraries(${PROJECT_NAME} PRIVATE nlohmann_json::nlohmann_json)

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

option(VERITAS_BUILD_PYBINDINGS "Build C++ to Python bindings" ON)
if (VERITAS_BUILD_PYBINDINGS)
    find_package(Python3 COMPONENTS Interpreter Development)
    add_subdirectory(lib/pybind11)
    pybind11_add_module("${PROJECT_NAME}_core" MODULE
        "${BINDINGS_DIR}/utility.cpp"
        "${BINDINGS_DIR}/interval.cpp"
        "${BINDINGS_DIR}/box.cpp"
        "${BINDINGS_DIR}/tree.cpp"
        "${BINDINGS_DIR}/addtree.cpp"
        "${BINDINGS_DIR}/featmap.cpp"
        "${BINDINGS_DIR}/search.cpp"
        "${BINDINGS_DIR}/bindings.cpp")
    target_link_libraries("${PROJECT_NAME}_core" PRIVATE ${PROJECT_NAME})
    install(TARGETS "${PROJECT_NAME}_core" DESTINATION "${SOURCE_DIR_PY}/${PROJECT_NAME}")
endif (VERITAS_BUILD_PYBINDINGS)

option(VERITAS_BUILD_CPPTESTS "Build C++ test files" OFF)
if (VERITAS_BUILD_CPPTESTS)
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
        "${TEST_DIR}/test_leafiter.cpp"
        "${TEST_DIR}/test_search.cpp"
        "${TEST_DIR}/test_memory.cpp"
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
    add_test(NAME test_leafiter
        COMMAND "test${PROJECT_NAME}" leafiter)
    add_test(NAME test_search
        COMMAND "test${PROJECT_NAME}" search)
    add_test(NAME test_memory
        COMMAND "test${PROJECT_NAME}" memory)
endif (VERITAS_BUILD_CPPTESTS)
