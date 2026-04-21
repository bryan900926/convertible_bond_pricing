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
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS ">> PROFILING DISABLED: Max speed mode (Release) <<")
    
    # Apply max speed flags ONLY in release
    target_compile_options(convertible_bond PRIVATE -O3 -march=native)
    
    # Disable Eigen's internal bounds checking for extra speed
    target_compile_definitions(convertible_bond PRIVATE NDEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS ">> DEBUG MODE: Unoptimized, with debug symbols <<")
    target_compile_options(convertible_bond PRIVATE -O0 -g)
endif()
target_link_libraries(convertible_bond PRIVATE Eigen3::Eigen)