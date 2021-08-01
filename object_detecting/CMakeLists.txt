cmake_minimum_required(VERSION 3.0.2)
project(run CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "-march=armv7-a")
set(CMAKE_BUILD_TYPE "Release")
add_executable(run object_detecting.cpp main.cpp)
find_package(InferenceEngine REQUIRED)
find_package(OpenCV REQUIRED)

target_link_libraries(run ${OpenCV_LIBRARIES} ${InferenceEngine_LIBRARIES})
