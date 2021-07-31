#include <gflags/gflags.h>
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include <map>
#include <opencv2/opencv.hpp>
#include <format_reader_ptr.h>
#include <inference_engine.hpp>
#include <ngraph/ngraph.hpp>

#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>

#include <vpu/vpu_tools_common.hpp>
#include <vpu/vpu_plugin_config.hpp>

#include "object_detection_sample_ssd.h"

using namespace InferenceEngine;

template <typename T>
void matU8ToBlob(const cv::Mat& orig_image, InferenceEngine::Blob::Ptr& blob, int batchIndex = 0) {
    InferenceEngine::SizeVector blobSize = blob->getTensorDesc().getDims();
    const size_t width = blobSize[3];
    const size_t height = blobSize[2];
    const size_t channels = blobSize[1];
    InferenceEngine::MemoryBlob::Ptr mblob = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
    if (!mblob) {
        THROW_IE_EXCEPTION << "We expect blob to be inherited from MemoryBlob in matU8ToBlob, "
            << "but by fact we were not able to cast inputBlob to MemoryBlob";
    }
    // locked memory holder should be alive all time while access to its buffer happens
    auto mblobHolder = mblob->wmap();

    T *blob_data = mblobHolder.as<T *>();

    cv::Mat resized_image(orig_image);
    if (static_cast<int>(width) != orig_image.size().width ||
            static_cast<int>(height) != orig_image.size().height) {
        cv::resize(orig_image, resized_image, cv::Size(width, height));
    }

   int image_size = 300 * 300;   
    for (size_t pid = 0; pid < image_size; pid++) {
        // Iterate over all channels 
        for (size_t ch = 0; ch < 3; ++ch) {
            //          [images stride + channels stride + pixel id ] all in bytes            
            blob_data[ch * image_size + pid] = ((float)resized_image.data[pid*3 + ch] - 127) / 128.0;
            //std::cout << blob_data[ch * image_size + pid] << " "; 
        }
        //std::cout << std::endl;
    }
}

bool ReadNetwork(InferenceEngine::Core &core, const std::string &modelName, InferenceEngine::CNNNetwork &network)
{
    bool ret = true;

    try
    { 
            // IR
            std::string binName = modelName.substr(0, modelName.length() - 4) + ".bin";
            network = core.ReadNetwork(modelName, binName);
    }
    catch (InferenceEngine::details::InferenceEngineException e)
    {
        printf("Error ReadNetwork() : %s\r\n", e.what());
        ret = false;
    }
    catch (...)
    {
        printf("Error ReadNetwork() \r\n");
        ret = false;
    }

    return ret;
}

bool LoadNetwork(const std::string& device, InferenceEngine::CNNNetwork &network, InferenceEngine::ExecutableNetwork &executableNetwork, InferenceEngine::Core &core)
{
    bool ret = true;

    try
    {
        executableNetwork = core.LoadNetwork(network, device);
    }
    catch (...)
    {
        ret = false;
        printf("Error : LoadNetwork()\r\n");
    }

    return ret;
}

InferenceEngine::InferRequest CreateInferRequest(InferenceEngine::ExecutableNetwork &executableNetwork)
{
    bool ret = true;
    InferenceEngine::InferRequest inferRequest;
    try
    {
        inferRequest = executableNetwork.CreateInferRequest();
    }
    catch (...)
    {
        ;
    }

    return inferRequest;
}

bool SetInputData(InferenceEngine::InferRequest &inferRequest,const std::string &imageName, std::string &inputLayerName)
{
    bool ret = true;
    cv::Mat imageData = cv::imread(imageName);

    InferenceEngine::Blob::Ptr inputBlob = inferRequest.GetBlob(inputLayerName);
    matU8ToBlob<float>(imageData, inputBlob);
    //inferRequest.SetBlob(inputLayerName, inputBlob);
    return ret;
}



int main(){
    InferenceEngine::Core core;
    InferenceEngine::CNNNetwork network;
    InferenceEngine::ExecutableNetwork executableNetwork;
    std::string model_path = "/home/pi/samples/build/mbv3-ssd-cornv1.xml";
    std::string device = "MYRIAD";
    std::string imagename = "/home/pi/samples/build/hikage_010_can.JPG";
    ReadNetwork(core, model_path, network);
    
    cv::Mat imageData = cv::imread(imagename);
    cv::resize(imageData, imageData, cv::Size(300, 300));

    InputsDataMap inputsInfo(network.getInputsInfo());
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
            Precision inputPrecision = Precision::FP32;
            item.second->setPrecision(inputPrecision);
            std::cout << item.second->getLayout() << " " << item.second->getPrecision() << std::endl;
            item.second->setLayout(InferenceEngine::Layout::NCHW);
            //item.second->getPreProcess().setColorFormat(InferenceEngine::ColorFormat::RGB);
        } else if (item.second->getInputData()->getTensorDesc().getDims().size() == 2) {
            //ここいらん
            std::cout << "no" << std::endl;
            imInfoInputName = item.first;

            Precision inputPrecision = Precision::FP32;
            item.second->setPrecision(inputPrecision);
            if ((item.second->getTensorDesc().getDims()[1] != 3 && item.second->getTensorDesc().getDims()[1] != 6)) {
                throw std::logic_error("Invalid input info. Should be 3 or 6 values length");
            }
        }
    }
    

    slog::info << "Preparing output blobs" << slog::endl;

    OutputsDataMap outputsInfo(network.getOutputsInfo());
    std::vector<std::string> output_names;
    std::vector<DataPtr> outputInfos;
    if (auto ngraphFunction = network.getFunction()) {
        for (const auto& out : outputsInfo) {
            output_names.push_back(out.first);
            DataPtr outputInfo = out.second;
            std::cout << outputInfo->getLayout() << " " << outputInfo->getPrecision() << std::endl;
            outputInfo->setPrecision(Precision::FP32);
            outputInfo->setLayout(InferenceEngine::Layout::CHW);
            outputInfos.push_back(outputInfo);
            const SizeVector outputDims = outputInfo->getTensorDesc().getDims();
            for(int i = 0; i < outputDims.size(); i++){
                std::cout << outputDims[i] << " ";
            }
            std::cout << std::endl;
        }
    }

    LoadNetwork(device, network, executableNetwork, core);
    InferenceEngine::InferRequest infer_request = CreateInferRequest(executableNetwork);
    std::cout << "SetInputData start" << std::endl;
    SetInputData(infer_request, imagename, imageInputName);
    std::cout << "SetInputData end" << std::endl;
    infer_request.Infer();
    infer_request.Wait(IInferRequest::WaitMode::RESULT_READY);
    std::cout << "Infer end" << std::endl;
    const float *output_concat = (infer_request.GetBlob(output_names[0]))->cbuffer().as<const PrecisionTrait<Precision::FP32>::value_type *>();
    const float *output_softmax = (infer_request.GetBlob(output_names[1]))->cbuffer().as<const PrecisionTrait<Precision::FP32>::value_type *>();
    std::vector<std::vector<int> > boxes;
    std::vector<std::vector<float> > labels;
    for(size_t i = 0; i < 3000; i++){
        std::vector<int> box;
        for(int l = 0; l < 4; l++){ //concat
            auto mid = static_cast<int>(output_concat[i * 4 + l] * 300);
            //std::cout << mid << " ";
            if(output_concat[i * 4 + l] > 2 && (i * 4 + l) < 9000)
                std::cout << i * 4 + l << " " <<  output_concat[i * 4 + l]  << std::endl;
            box.push_back(mid);
        }
        //std::cout << std::endl;
        boxes.push_back(box);
        std::vector<float> label;
        for(int l = 0; l < 3; l++){ //softmax
            auto mid = output_softmax[i * 3 + l];
            label.push_back(mid);
        }
        labels.push_back(label);            
    }

    

    double threshold = 0.2;
    for(size_t i = 0; i < 3000; i++){
        for(int l = 1; l < 3; l++){
            if(labels[i][l] > threshold){
                /*
                if(boxes[i][3] < 0 || boxes[i][2] < 0){
                    continue;
                }
                */
                std::cout << labels[i][l] << std::endl;
                int xmin = std::max(0, boxes[i][0]);
                int ymin = std::max(0, boxes[i][1]);
                int xmax = std::min(300, boxes[i][2]);
                int ymax = std::min(300, boxes[i][3]);
                
                //if(xmin > 300 || ymin > 300 || xmax < 0 || ymax < 0)
                std::cout << xmin << " " << ymin << " " << xmax << " " << ymax << std::endl;
                
                cv::rectangle(imageData, cv::Point(xmin,ymin), cv::Point(xmax,ymax), cv::Scalar(255,0,0), 2);
            }
        }
    }
    cv::imwrite("./result.png", imageData);
}