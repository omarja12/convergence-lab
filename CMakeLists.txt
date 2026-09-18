cmake_minimum_required(VERSION 3.16)
project(convergence-lab LANGUAGES CXX VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Header-only library: everything lives in include/bspde.
add_library(bspde INTERFACE)
target_include_directories(bspde INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

if(MSVC)
  target_compile_options(bspde INTERFACE /W4 /permissive- /Zc:__cplusplus)
else()
  target_compile_options(bspde INTERFACE -Wall -Wextra -Wpedantic)
endif()

add_executable(bspde_demo src/main.cpp)
target_link_libraries(bspde_demo PRIVATE bspde)

enable_testing()
add_executable(bspde_tests tests/test_convergence.cpp)
target_link_libraries(bspde_tests PRIVATE bspde)
add_test(NAME convergence COMMAND bspde_tests)
