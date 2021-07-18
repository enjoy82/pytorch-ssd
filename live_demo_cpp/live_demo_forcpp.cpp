//from https://qiita.com/fan2tamo/items/36bc8f9657d1a430aa54#8-process-output
#include <iostream>
#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>
#include <ie_iextension.h>
//#include <ext_list.hpp>
#include <string>

using namespace InferenceEngine;

bool LoadPlugin(const std::string& device, InferencePlugin& plugin)
{
    bool ret = true;

    try
    {
        plugin = PluginDispatcher().getPluginByDevice(device);

        if (device == "CPU")
        {
            //plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
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
        //ここ死んでる
        input_info = InputsDataMap(network.getInputsInfo());

        for (auto&& input : input_info)
        {
            input_name = input.first;
            input.second->setPrecision(precision);
            std::cout << "configure input start" << std::endl;
            input.second->setLayout(layout);
            std::cout << "configure input start2" << std::endl;
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
    std::string device = "GPU";
    //std::string device = "MYRIAD";
    std::string modelPath = "./models/mbv3-ssd-cornv1.xml";
    int result = 0;


    //set up camera
    int camera_id = 0;
    double fps = 30.0;
    double width = 640.0;
    double height = 480.0;
    float threshold = 0.5;
    cv::VideoCapture cap(cv::CAP_DSHOW + camera_id);
    if(!cap.isOpened()){ //エラー処理
		 std::cout << "cap error" << std::endl;
		 return -1;
	}
    if (!cap.set(cv::CAP_PROP_FPS, fps)) std::cout << "camera set fps error" << std::endl;
    if (!cap.set(cv::CAP_PROP_FRAME_WIDTH, width)) std::cout << "camera set width error" << std::endl;
    if (!cap.set(cv::CAP_PROP_FRAME_HEIGHT, height)) std::cout << "camera set height error" << std::endl;
    cv::Mat frame;
    
    //TODO refactor
    LoadPlugin(device, plugin);
    ReadModel(modelPath, network);
    ConfigureInput(network, input_info, input_name, Precision::U8, Layout::NCHW);
    std::cout << "configure input OK" << std::endl;
    ConfigureOutput(network, output_info, output_name, Precision::FP32, Layout::NC);
    std::cout << "configure output OK" << std::endl;
    LoadModel(network, plugin, executable_network);
    std::cout << "LoadModel OK" << std::endl;
    CreateInferRequest(executable_network, async_infer_request);
    std::cout << "CreateInferRequest OK" << std::endl;
    DataPtr& output = output_info.begin()->second;
    const SizeVector outputDims = output->getTensorDesc().getDims();
    const int numDetections = outputDims[2];
    const int objectSize = outputDims[3];

    while(cap.read(frame)){
        cv::imshow("frame", frame);
        int key = cv::waitKey(1);
		if(key == 'q'){
			cv::destroyWindow("frame");
			break;
        }
        
        PrepareInput(async_infer_request, input_name, frame);
        Infer(async_infer_request);
        //result = ProcessOutput(async_infer_request, output_name);
        cv::putText(frame, "test", cv::Point2f(0, 20), cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 0, 255));
        const float *detections = async_infer_request->GetBlob(output_name)->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
        //TODO class classification
        for (int i = 0; i < numDetections; i++) {
            float confidence = detections[i * objectSize + 2];
            float xmin = detections[i * objectSize + 3] * width;
            float ymin = detections[i * objectSize + 4] * height;
            float xmax = detections[i * objectSize + 5] * width;
            float ymax = detections[i * objectSize + 6] * height;

            if (confidence > threshold) {
                std::ostringstream conf;
                conf << std::fixed << std::setprecision(3) << confidence;
                cv::putText(frame,
                    conf.str(),
                    cv::Point2f(xmin, ymin - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1,
                    cv::Scalar(0, 0, 255));
                cv::rectangle(frame, cv::Point2f(xmin, ymin), cv::Point2f(xmax, ymax), cv::Scalar(0, 0, 255));
            }
        }

        cv::imshow("object Detector", frame);
    }
}
