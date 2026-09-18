cmake_minimum_required(VERSION 3.16)
project(convergence-lab LANGUAGES CXX VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Header-only library: everything lives in include/convergence.
add_library(convergence INTERFACE)
target_include_directories(convergence INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

if(MSVC)
  target_compile_options(convergence INTERFACE /W4 /permissive- /Zc:__cplusplus)
else()
  target_compile_options(convergence INTERFACE -Wall -Wextra -Wpedantic)
endif()

add_executable(convergence_demo src/main.cpp)
target_link_libraries(convergence_demo PRIVATE convergence)

# Emits docs/convergence.csv, the data the project page is plotted from.
add_executable(convergence_study tools/convergence_study.cpp)
target_link_libraries(convergence_study PRIVATE convergence)

enable_testing()
add_executable(convergence_tests tests/test_convergence.cpp)
target_link_libraries(convergence_tests PRIVATE convergence)
add_test(NAME convergence COMMAND convergence_tests)
