cmake_minimum_required(VERSION 3.15)

project(convertible_bond LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. FetchContent MUST be included before you declare it
include(FetchContent)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
)
FetchContent_MakeAvailable(eigen)

# 2. Find OpenMP
find_package(OpenMP REQUIRED)

file(GLOB_RECURSE MY_SOURCES "src/*.cpp")
file(GLOB_RECURSE MY_HEADERS "src/*.hpp")


add_executable(convertible_bond
    main.cpp
    ${MY_SOURCES}
)

target_include_directories(convertible_bond PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# 3. Link Eigen (Wait to link OpenMP below)
target_link_libraries(convertible_bond PRIVATE 
    Eigen3::Eigen
)

# 4. The Bulletproof OpenMP Fix: 
# Instead of relying on the buggy OpenMP::OpenMP_CXX magic target, 
# we inject the raw compiler flags directly. This works on every compiler.
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
    message(STATUS ">> RELEASE MODE: Max speed, no safety checks <<")
    target_compile_options(convertible_bond PRIVATE -O3)
    target_compile_definitions(convertible_bond PRIVATE NDEBUG)

elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    message(STATUS ">> REL_WITH_DEB_INFO: Max speed, but KEEP debug symbols <<")
    target_compile_options(convertible_bond PRIVATE 
            -O3 
            # -march=native 
            # -fsanitize=address
            -fstack-protector-all # Force the compiler to check for stack smashing 
            -g 
            -fno-omit-frame-pointer
        )

elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS ">> DEBUG MODE: Unoptimized, with debug symbols <<")
    target_compile_options(convertible_bond PRIVATE -O0 -g)
endif()