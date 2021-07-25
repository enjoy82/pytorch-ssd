cmake_minimum_required(VERSION 3.0.2)
project(run CXX)

add_executable(run sample_copy.cpp)
find_package(InferenceEngine 2.1 REQUIRED)
find_package(ngraph REQUIRED)
find_package(OpenCV REQUIRED)

target_link_libraries(run ${NGRAPH_LIBRARIES} ${OpenCV_LIBRARIES} ${InferenceEngine_LIBRARIES})
