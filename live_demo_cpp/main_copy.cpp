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


using namespace InferenceEngine;

/**
* \brief The entry point for the Inference Engine object_detection sample application
* \file object_detection_sample_ssd/main.cpp
* \example object_detection_sample_ssd/main.cpp
*/
int main(int argc, char *argv[]) {
    Core ie;
    int camera_id = 0;
    double fps = 30.0;
    double width = 320.0;
    double height = 240.0;
    double input_width = 300.0;
    double input_height = 300.0;
    float threshold = 0.5;
    std::string device = "MYRIAD";
    //std::string device = "CPU";
    std::string modelPath = "/home/pi/pytorch-ssd/live_demo_cpp/models/mbv3-ssd-cornv1.xml";
    std::string binPath = "/home/pi/pytorch-ssd/live_demo_cpp/models/mbv3-ssd-cornv1.bin";
    

    CNNNetwork network = ie.ReadNetwork(modelPath);
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 5. Prepare input blobs --------------------------------------------------
    std::cout << "Preparing input blobs" << std::endl;

    /** Taking information about all topology inputs **/
    InputsDataMap inputsInfo(network.getInputsInfo());
    
    /** SSD network has one input and one output **/
    if (inputsInfo.size() != 1 && inputsInfo.size() != 2) throw std::logic_error("Sample supports topologies only with 1 or 2 inputs");

    /**
     * Some networks have SSD-like output format (ending with DetectionOutput layer), but
     * having 2 inputs as Faster-RCNN: one for image and one for "image info".
     *
     * Although object_datection_sample_ssd's main task is to support clean SSD, it could score
     * the networks with two inputs as well. For such networks imInfoInputName will contain the "second" input name.
     */
    std::string imageInputName, imInfoInputName;

    InputInfo::Ptr inputInfo = nullptr;

    SizeVector inputImageDims;
    /** Stores input image **/

    /** Iterating over all input blobs **/
    for (auto & item : inputsInfo) {
        /** Working with first input tensor that stores image **/
        if (item.second->getInputData()->getTensorDesc().getDims().size() == 4) {
            imageInputName = item.first;

            inputInfo = item.second;

            std::cout << "Batch size is " << std::to_string(network.getBatchSize()) << std::endl;

            /** Creating first input blob **/
            Precision inputPrecision = Precision::U8;
            item.second->setPrecision(inputPrecision);
        } else if (item.second->getInputData()->getTensorDesc().getDims().size() == 2) {
            std::cout << "FP32" << std::endl;
            imInfoInputName = item.first;

            Precision inputPrecision = Precision::FP32;
            item.second->setPrecision(inputPrecision);
            if ((item.second->getTensorDesc().getDims()[1] != 3 && item.second->getTensorDesc().getDims()[1] != 6)) {
                throw std::logic_error("Invalid input info. Should be 3 or 6 values length");
            }
        }
    }

    if (inputInfo == nullptr) {
        inputInfo = inputsInfo.begin()->second;
    }
    // ---------------------------------------------
    std::cout << "set output info" << std::endl;
    OutputsDataMap outputsInfo(network.getOutputsInfo());
    std::vector<std::string> output_names; //first is concat, second is softmax
    std::vector<int> numDetections;
    std::vector<int> objectSizes;
    for (auto &item : outputsInfo) {
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
    std::cout << "prepare OK" << std::endl;
    return 0;
}
