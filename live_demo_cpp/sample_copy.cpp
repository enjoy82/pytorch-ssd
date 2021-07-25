// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include <map>

#include <inference_engine.hpp>
#include <ngraph/ngraph.hpp>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <ie_core.hpp>
#include<fstream>
#include <algorithm>
//#include <ext_list.hpp>
#include <string>
#include <map>

using namespace InferenceEngine;

bool checkFileExistence(const std::string& str)
{
    std::ifstream ifs(str);
    return ifs.is_open();
}

InferenceEngine::Blob::Ptr wrapMat2Blob(const cv::Mat &mat) {
    size_t channels = mat.channels();
    size_t height = mat.size().height;
    size_t width = mat.size().width;
    size_t strideH = mat.step.buf[0];
    size_t strideW = mat.step.buf[1];

    bool is_dense =
            strideW == channels &&
            strideH == channels * width;

    InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::U8,
                                      {1, channels, height, width},
                                      InferenceEngine::Layout::NHWC);

    return InferenceEngine::make_shared_blob<uint8_t>(tDesc, mat.data);
}


/**
* \brief The entry point for the Inference Engine object_detection sample application
* \file object_detection_sample_ssd/main.cpp
* \example object_detection_sample_ssd/main.cpp
*/
int main(int argc, char *argv[]) {
    std::string input_name;
    std::string device = "MYRIAD";
    std::string modelPath = "/home/pi/pytorch-ssd/live_demo_cpp/models/mbv3-ssd-cornv1.xml";
    std::string binPath = "/home/pi/pytorch-ssd/live_demo_cpp/models/mbv3-ssd-cornv1.bin";
    /*
    int camera_id = 0;
    double fps = 30.0;
    double width = 320.0;
    double height = 240.0;
    double input_width = 300.0;
    double input_height = 300.0;
    float threshold = 0.5;
    if(!cap.isOpened()){ //エラー処理
        std::cout << "cap error" << std::endl;
        return -1;
    }
    if (!cap.set(cv::CAP_PROP_FPS, fps)) std::cout << "camera set fps error" << std::endl;
    if (!cap.set(cv::CAP_PROP_FRAME_WIDTH, width)) std::cout << "camera set width error" << std::endl;
    if (!cap.set(cv::CAP_PROP_FRAME_HEIGHT, height)) std::cout << "camera set height error" << std::endl;
    cv::Mat frame;
    */
    if(checkFileExistence(modelPath) && checkFileExistence(binPath)){
        std::cout << "model path exist" << std::endl;
    }else{
        std::cout << "model path error!" << std::endl;
    }
    int result = 0;
    Core ie;

/** Read network model **/
    CNNNetwork network = ie.ReadNetwork(modelPath);
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 5. Prepare input blobs --------------------------------------------------
    std::cout << "Preparing input blobs" << std::endl;

    /** Taking information about all topology inputs **/
    InferenceEngine::InputsDataMap input_info(network.getInputsInfo());
    InferenceEngine::OutputsDataMap output_info(network.getOutputsInfo());

        std::cout << "configure input end" << std::endl;

    for (auto &item : input_info) {
        input_name = item.first;
        auto input_data = item.second;
        input_data->setPrecision(Precision::U8);
        input_data->setLayout(Layout::NCHW);
        input_data->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
        input_data->getPreProcess().setColorFormat(ColorFormat::RGB);
    }
    std::vector<std::string> output_names; //first is concat, second is softmax
    std::vector<int> numDetections;
    std::vector<int> objectSizes;
    for (auto &item : output_info) {
        output_names.push_back(item.first);
        auto output_data = item.second;
        output_data->setPrecision(Precision::FP32);
        output_data->setLayout(Layout::CHW);
        const SizeVector outputDims = output_data->getTensorDesc().getDims();
        numDetections.push_back(outputDims[1]);
        objectSizes.push_back(outputDims[2]);
    }
    for(int i = 0; i < output_names.size(); i++){
        std::cout << output_names[i] << " " << numDetections[i] << " " << objectSizes[i] << std::endl;
    }    //ConfigureOutput(network, output_info, output_name, Precision::FP32, Layout::NC);
    std::cout << "configure output end" << std::endl;


    // -----------------------------------------------------------------------------------------------------
    std::map<std::string, std::string> config = {};
    ExecutableNetwork executable_network = ie.LoadNetwork(network, device, config);
    std::cout << "loadnetwork OK" << std::endl;

    return 0;
}
