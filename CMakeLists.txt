cmake_minimum_required(VERSION 3.15)

project(convertible_bond LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0  # Use a stable version tag
)

FetchContent_MakeAvailable(Eigen)

file(GLOB_RECURSE MY_SOURCES "src/*.cpp")

add_executable(convertible_bond
    main.cpp
    ${MY_SOURCES}
)

target_include_directories(convertible_bond PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Wunused-variable
    -Wunused-parameter
)

find_package(OpenMP REQUIRED)

if(OpenMP_CXX_FOUND)
    target_link_libraries(convertible_bond PUBLIC OpenMP::OpenMP_CXX)
endif()

# --- NEW ---
option(ENABLE_PROFILING "Enable gprof profiling" OFF) # Defaults to OFF

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND ENABLE_PROFILING)
    message(STATUS ">> PROFILING ENABLED: Code will be slower but trackable <<")
    target_compile_options(convertible_bond PRIVATE -pg)
    target_link_options(convertible_bond PRIVATE -pg -no-pie)
else()
    message(STATUS ">> PROFILING DISABLED: Max speed mode <<")
    # Add Optimization flags for Release mode
    target_compile_options(convertible_bond PRIVATE -O3 -march=native)
endif()

target_link_libraries(convertible_bond PRIVATE Eigen3::Eigen)