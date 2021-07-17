cmake_minimum_required(VERSION 3.0.2)
project(run CXX)
set(TARGET_NAME "livedemo")

add_executable(${TARGET_NAME} live_demo_forcpp.cpp)
find_package(InferenceEngine 2.1 REQUIRED)
find_package(OpenCV REQUIRED)

target_link_libraries(${TARGET_NAME} PRIVATE ${OpenCV_LIBRARIES} ${InferenceEngine_LIBRARIES})