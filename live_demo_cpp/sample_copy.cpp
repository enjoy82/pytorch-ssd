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


/**
* \brief The entry point for the Inference Engine object_detection sample application
* \file object_detection_sample_ssd/main.cpp
* \example object_detection_sample_ssd/main.cpp
*/
int main(int argc, char *argv[]) {
    try {
        //CNNNetReader network_reader;
        std::string input_name;
        //std::string device = "GPU";
        std::string device = "MYRIAD";
        //std::string device = "CPU";
        std::string modelPath = "/home/pi/pytorch-ssd/live_demo_cpp/models/mbv3-ssd-cornv1.xml";
        std::string binPath = "/home/pi/pytorch-ssd/live_demo_cpp/models/mbv3-ssd-cornv1.bin";
        if(checkFileExistence(modelPath) && checkFileExistence(binPath)){
            std::cout << "model path exist" << std::endl;
        }else{
            std::cout << "model path error!" << std::endl;
        }
        std::cout << core.GetVersions(device) << std::endl;
        int result = 0;
        Core ie;

	/** Read network model **/
        CNNNetwork network = ie.ReadNetwork(modelPath);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Prepare input blobs --------------------------------------------------
        slog::info << "Preparing input blobs" << slog::endl;

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

                slog::info << "Batch size is " << std::to_string(network.getBatchSize()) << slog::endl;

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
        // -----------------------------------------------------------------------------------------------------
	    std::map<std::string, std::string> config = {};
        ExecutableNetwork executable_network = ie.LoadNetwork(network, FLAGS_d, config);
        std::cout << "loadnetwork OK" << std::endl;

    return 0;
}
