#pragma once
//#ifndef ROVER_NO_OPENCV
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
/*
#else
namespace cv
{
    class Mat;
}
#endif
*/
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
/*
struct ObjectDetectionResult
{

}
*/
class ObjectDetector
{
private:
    InferenceEngine::Core core;
    InferenceEngine::CNNNetwork network;
    InferenceEngine::ExecutableNetwork executableNetwork;
    InferenceEngine::InferRequest inferRequest;
    std::string inputName;
    //今回構築したモデルが2つoutputを返すのでvector
    std::vector<std::string> outputName;
public:
    explicit ObjectDetector();

    ~ObjectDetector() = default;

    bool readNetwork(const std::string &modelPath);

    bool setInputInfos();

    bool setOutputInfos();

    bool loadNetwork(std::string &device);

    bool createInferRequest();

    bool initObjectDetector(std::string &modelPath, std::string &device);

    bool setInputData(cv::Mat &image);

    void matU8ToBlobFP32(const cv::Mat &orgImage, InferenceEngine::Blob::Ptr &blob);

    bool infer();

    //ここ何返すか悩む
    std::vector<std::vector<std::vector<float> > > getOutputData(double threshold);
};