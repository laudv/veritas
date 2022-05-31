# https://github.com/benjaminjack/python_cpp_example

cmake_minimum_required(VERSION 3.1)
project(veritas)

set(SOURCE_DIR "src/cpp")
# Tell cmake that headers are also in source_dir
include_directories(${SOURCE_DIR})
set(SOURCES
    "${SOURCE_DIR}/tree.cpp"
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
find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)

option(BUILD_PYTHON_BINDINGS "Build C++ to Python bindings" ON)
if (BUILD_PYTHON_BINDINGS)
    #find_package(pybind11 REQUIRED)
    add_subdirectory(lib/pybind11)
    pybind11_add_module("py${PROJECT_NAME}" "${SOURCE_DIR}/bindings.cpp")
    target_link_libraries("py${PROJECT_NAME}" PRIVATE ${PROJECT_NAME})

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
endif (BUILD_TESTS)
