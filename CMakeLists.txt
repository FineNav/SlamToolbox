cmake_minimum_required(VERSION 3.18)
project(slam_toolbox_native LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(octomap REQUIRED)

pybind11_add_module(
    _native
    cpp/src/bindings/module.cpp
    cpp/src/bindings/local_hash_voxel_bindings.cpp
    cpp/src/bindings/raycast_voxel_bindings.cpp
    cpp/src/local_hash_voxel/engine.cpp
    cpp/src/raycast_voxel/engine.cpp
)

target_include_directories(_native PRIVATE cpp/include)
target_compile_features(_native PRIVATE cxx_std_17)

if(TARGET octomap-static AND TARGET octomath-static)
    target_link_libraries(_native PRIVATE octomap-static octomath-static)
else()
    target_include_directories(_native PRIVATE ${OCTOMAP_INCLUDE_DIRS})
    target_link_libraries(_native PRIVATE ${OCTOMAP_LIBRARIES})
endif()

if(MSVC)
    target_compile_options(_native PRIVATE /O2 /W4)
else()
    target_compile_options(_native PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

install(TARGETS _native LIBRARY DESTINATION slam_toolbox)
