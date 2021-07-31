cmake_minimum_required(VERSION 3.0.2)
project(run CXX)

add_executable(run main3.cpp)
find_package(InferenceEngine 2.1 REQUIRED)
find_package(OpenCV REQUIRED)

target_link_libraries(run ${OpenCV_LIBRARIES} ${InferenceEngine_LIBRARIES})
