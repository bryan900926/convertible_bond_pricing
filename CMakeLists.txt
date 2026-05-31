cmake_minimum_required(VERSION 3.15)

project(convertible_bond LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(Eigen3 3.4 REQUIRED)
find_package(OpenMP REQUIRED)

file(GLOB_RECURSE MY_SOURCES "src/*.cpp")

add_executable(convertible_bond
    main.cpp
    ${MY_SOURCES}
    src/Equity/EquityTreeBuildMemoSave.cpp
    src/Util/TreeManager.hpp
)

target_include_directories(convertible_bond PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(convertible_bond PRIVATE 
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)

add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Wunused-variable
    -Wunused-parameter
)


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
    # target_link_options(convertible_bond PRIVATE 
    #                     -fsanitize=address 
    # )

elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS ">> DEBUG MODE: Unoptimized, with debug symbols <<")
    target_compile_options(convertible_bond PRIVATE -O0 -g)
endif()