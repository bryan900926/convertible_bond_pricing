cmake_minimum_required(VERSION 3.15)

project(convertible_bond LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
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
    src/Equity/EquityTreeBuildMemoSave.cpp
)

target_include_directories(convertible_bond PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_include_directories(convertible_bond PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/boost/boost_1_91_0
)

add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Wunused-variable
    -Wunused-parameter
)

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(convertible_bond PRIVATE -fopenmp)
    target_link_options(convertible_bond PRIVATE -fopenmp)
elseif(MSVC)
    target_compile_options(convertible_bond PRIVATE /openmp)
else()
    target_link_libraries(convertible_bond PRIVATE OpenMP::OpenMP_CXX)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS ">> RELEASE MODE: Max speed, no safety checks <<")
    target_compile_options(convertible_bond PRIVATE -O3 -march=native)
    target_compile_definitions(convertible_bond PRIVATE NDEBUG)

elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    message(STATUS ">> REL_WITH_DEB_INFO: Max speed, but KEEP debug symbols <<")
    target_compile_options(convertible_bond PRIVATE -O3 -march=native -g)
    # Notice we DO NOT define NDEBUG here. We want Eigen to crash and tell us where it failed!

elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS ">> DEBUG MODE: Unoptimized, with debug symbols <<")
    target_compile_options(convertible_bond PRIVATE -O0 -g)
endif()
target_link_libraries(convertible_bond PRIVATE Eigen3::Eigen)