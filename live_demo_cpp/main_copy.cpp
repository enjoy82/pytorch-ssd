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
    InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;
    std::string input_name = network.getInputsInfo().begin()->first;
    input_info->setLayout(Layout::NHWC);
    input_info->setPrecision(Precision::U8);
    /** SSD network has one input and one output **/

    /**
     * Some networks have SSD-like output format (ending with DetectionOutput layer), but
     * having 2 inputs as Faster-RCNN: one for image and one for "image info".
     *
     * Although object_datection_sample_ssd's main task is to support clean SSD, it could score
     * the networks with two inputs as well. For such networks imInfoInputName will contain the "second" input name.
     */
    DataPtr output_info1 = network.getOutputsInfo().begin()->second;
    std::string output_name1 = network.getOutputsInfo().begin()->first;

    DataPtr output_info2 = network.getOutputsInfo().end()->second;
    std::string output_name2 = network.getOutputsInfo().end()->first;
    std::cout << output_name1 << " " << output_name << std::endl;
    // ---------------------------------------------
    /*
    std::cout << "set output info" << std::endl;
    OutputsDataMap outputsInfo(network.getOutputsInfo());
    std::vector<std::string> output_names; //first is concat, second is softmax
    std::vector<int> numDetections;
    std::vector<int> objectSizes;
    if (auto ngraphFunction = network.getFunction()) {
        for (const auto& out : outputsInfo) {
            for (const auto & op : ngraphFunction->get_ops()) {
                if (op->get_type_info() == ngraph::op::DetectionOutput::type_info &&
                        op->get_friendly_name() == out.second->getName()) {
                    output_names.push_back(out.first);
                    auto output_data = out.second;
                    const SizeVector outputDims = output_data->getTensorDesc().getDims();
                    numDetections.push_back(outputDims[1]);
                    objectSizes.push_back(outputDims[2]);
                    break;
                }
            }
        }
    } 
    for(int i = 0; i < output_names.size(); i++){
        std::cout << output_names[i] << " " << numDetections[i] << " " << objectSizes[i] << std::endl;
    }
    */
    std::cout << "configure output end" << std::endl;
    

    // -----------------------------------------------------------------------------------------------------
	std::map<std::string, std::string> config = {};
    ExecutableNetwork executable_network = ie.LoadNetwork(network, device, config);
    std::cout << "loadnetwork OK" << std::endl;
    std::cout << "prepare OK" << std::endl;
    return 0;
}
