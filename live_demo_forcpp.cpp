#include <iostream>
#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>
#include <ie_iextension.h>
#include <ext_list.hpp>

using namespace InferenceEngine;

bool LoadPlugin(const std::string& device, InferencePlugin& plugin)
{
    bool ret = true;

    try
    {
        plugin = PluginDispatcher().getPluginByDevice(device);

        if (device == "CPU")
        {
            plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
        }
    }
    catch (const std::exception & ex)
    {
        OutputDebugStringA(ex.what());
        ret = false;
    }

    return ret;
}

bool ReadModel(const std::string &modelPath, CNNNetwork& network)
{
    bool ret = true;
    CNNNetReader network_reader;

    try
    {
        network_reader.ReadNetwork(modelPath);
        network_reader.ReadWeights(modelPath.substr(0, modelPath.size() - 4) + ".bin");
        network_reader.getNetwork().setBatchSize(1);
        network = network_reader.getNetwork();
    }
    catch (const std::exception & ex)
    {
        OutputDebugStringA(ex.what());
        ret = false;
    }

    return ret;
}

bool ConfigureInput(CNNNetwork& network, InputsDataMap& input_info, std::string& input_name, const Precision precision, const Layout layout)
{
    bool ret = true;

    try
    {
        input_info = InputsDataMap(network.getInputsInfo());

        for (auto&& input : input_info)
        {
            input_name = input.first;
            input.second->setPrecision(precision);
            input.second->setLayout(layout);
        }
    }
    catch (const std::exception & ex)
    {
        OutputDebugStringA(ex.what());
        ret = false;
    }

    return ret;
}

bool ConfigureOutput(CNNNetwork& network, OutputsDataMap& output_info, std::string& output_name, const Precision precision, const Layout layout)
{
    bool ret = true;

    try
    {
        output_info = OutputsDataMap(network.getOutputsInfo());

        for (auto&& output : output_info)
        {
            output_name = output.first;
            output.second->setPrecision(precision);
            output.second->setLayout(layout);
        }
    }
    catch (const std::exception & ex)
    {
        OutputDebugStringA(ex.what());
        ret = false;
    }

    return ret;
}

bool LoadModel(CNNNetwork& network, InferencePlugin& plugin, ExecutableNetwork& executable_network)
{
    bool ret = true;

    try
    {
        executable_network = plugin.LoadNetwork(network, {});
    }
    catch (const std::exception & ex)
    {
        OutputDebugStringA(ex.what());
        ret = false;
    }

    return ret;
}

bool CreateInferRequest(ExecutableNetwork& executable_network, InferRequest::Ptr& async_infer_request)
{
    bool ret = true;

    try
    {
        async_infer_request = executable_network.CreateInferRequestPtr();
    }
    catch (const std::exception & ex)
    {
        OutputDebugStringA(ex.what());
        ret = false;
    }

    return ret;
}

template <typename T>
void matU8ToBlob(const cv::Mat& orig_image, InferenceEngine::Blob::Ptr& blob, int batchIndex = 0)
{
    InferenceEngine::SizeVector blobSize = blob->getTensorDesc().getDims();
    const size_t width = blobSize[3];
    const size_t height = blobSize[2];
    const size_t channels = blobSize[1];
    T* blob_data = blob->buffer().as<T*>();

    cv::Mat resized_image(orig_image);
    if (width != orig_image.size().width || height != orig_image.size().height) {
        cv::resize(orig_image, resized_image, cv::Size(width, height));
    }

    int batchOffset = batchIndex * width * height * channels;

    for (size_t c = 0; c < channels; c++) {
        for (size_t h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                blob_data[batchOffset + c * width * height + h * width + w] = resized_image.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
}

bool PrepareInput(InferRequest::Ptr& async_infer_request, const std::string & input_name, const cv::Mat & image)
{
    bool ret = true;

    try
    {
        Blob::Ptr input = async_infer_request->GetBlob(input_name);
        matU8ToBlob<uint8_t>(image, input);
    }
    catch (const std::exception & ex)
    {
        OutputDebugStringA(ex.what());
        ret = false;
    }

    return ret;
}

bool Infer(InferRequest::Ptr& async_infer_request)
{
    bool ret = true;

    try
    {
        async_infer_request->StartAsync();
        async_infer_request->Wait(IInferRequest::WaitMode::RESULT_READY);
    }
    catch (const std::exception & ex)
    {
        OutputDebugStringA(ex.what());
        ret = false;
    }

    return ret;
}

//TODO 書き換え
int ProcessOutput(InferRequest::Ptr& async_infer_request, const std::string& output_name)
{

    int result = 0;
    float buf= 0;

    try
    {
        const float* oneHotVector = async_infer_request->GetBlob(output_name)->buffer().as<float*>();

        for (int i = 0; i < 10; i++)
        {
            if (oneHotVector[i] > buf)
            {
                buf = oneHotVector[i];
                result = i;
            }
        }
    }
    catch (const std::exception & ex)
    {
        OutputDebugStringA(ex.what());
        result = -1;
    }

    return result;
}

int main(){
    InferencePlugin plugin;
    CNNNetwork network;
    InputsDataMap input_info;
    OutputsDataMap output_info;
    ExecutableNetwork executable_network;
    InferRequest::Ptr async_infer_request;
    std::string input_name;
    std::string output_name;
    std::string device = "CPU"
    std::string modelPath = "test.xml"
    std::string imagePath = "test.jpg";
    int result = 0;

    cv::Mat img = cv::imread(imagePath, 1);

    LoadPlugin(device, plugin);
    ReadModel(modelPath, network);
    ConfigureInput(network, input_info, input_name, Precision::U8, Layout::NCHW);
    ConfigureOutput(network, output_info, output_name, Precision::FP32, Layout::NC);
    LoadModel(network, plugin, executable_network);
    CreateInferRequest(executable_network, async_infer_request);
    PrepareInput(async_infer_request, input_name, img);
    Infer(async_infer_request);
    result = ProcessOutput(async_infer_request, output_name);

    printf("result = %d", result);
}
