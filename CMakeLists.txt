cmake_minimum_required(VERSION 3.15)

project(convertible_bond LANGUAGES CXX)

add_compile_options(-pipe)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. FetchContent MUST be included before you declare it
include(FetchContent)
set(BUILD_TESTING OFF CACHE BOOL "Disable tests" FORCE)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
)
FetchContent_MakeAvailable(eigen)

# 1. Declare where to get RapidCSV
FetchContent_Declare(
  rapidcsv
  GIT_REPOSITORY https://github.com/d99kris/rapidcsv.git
  GIT_TAG        v8.84 # You can use "master" here, but a specific version tag is safer
)

FetchContent_MakeAvailable(rapidcsv)

FetchContent_Declare(
    boost_pfr
    URL "https://github.com/boostorg/pfr/archive/refs/heads/master.zip"
)

FetchContent_MakeAvailable(boost_pfr)

# 2. Find OpenMP
find_package(OpenMP REQUIRED)

file(GLOB_RECURSE MY_SOURCES "src/*.cpp")
file(GLOB_RECURSE MY_HEADERS "src/*.hpp")


add_executable(convertible_bond
    main.cpp
    ${MY_SOURCES}
    src/Const/ParameterColumn.hpp
)

target_include_directories(convertible_bond PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_include_directories(convertible_bond PRIVATE 
    ${boost_pfr_SOURCE_DIR}/include
)

# 3. Link Eigen (Wait to link OpenMP below)
target_link_libraries(convertible_bond PRIVATE 
    Eigen3::Eigen
)
target_include_directories(convertible_bond PRIVATE ${rapidcsv_SOURCE_DIR}/src)


target_compile_options(convertible_bond PRIVATE ${OpenMP_CXX_FLAGS})
target_link_options(convertible_bond PRIVATE ${OpenMP_CXX_FLAGS})


add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Wunused-variable
    -Wunused-parameter
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS ">> REL_WITH_DEB_INFO: Max speed, but KEEP debug symbols <<")
    target_compile_options(convertible_bond PRIVATE 
            -O2 
            -fstack-protector-all # Force the compiler to check for stack smashing 
            -g 
            -fno-omit-frame-pointer
        )

elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS ">> DEBUG MODE: Unoptimized, with debug symbols <<")
    target_compile_options(convertible_bond PRIVATE -O0 -g)
endif()