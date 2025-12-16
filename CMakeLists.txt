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

target_link_libraries(convertible_bond PRIVATE Eigen3::Eigen)