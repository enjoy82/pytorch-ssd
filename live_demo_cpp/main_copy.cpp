// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gflags/gflags.h>
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include <map>

#include <format_reader_ptr.h>
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
    std::map<std::string, std::string> config = {};
    // -----------------------------------------------------------------------------------------------------
	ExecutableNetwork executable_network = ie.LoadNetwork(network, device, config);
    std::cout << "loadnetwork OK" << std::endl;
    std::cout << "prepare OK" << std::endl;
    return 0;
}
