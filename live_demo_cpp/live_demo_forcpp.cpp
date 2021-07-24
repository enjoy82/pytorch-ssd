//from https://qiita.com/fan2tamo/items/36bc8f9657d1a430aa54#8-process-output
#include <iostream>
#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>
#include <ie_iextension.h>
#include<fstream>

//#include <ext_list.hpp>
#include <string>

using namespace InferenceEngine;

bool checkFileExistence(const std::string& str)
{
    std::ifstream ifs(str);
    return ifs.is_open();
}

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
        //OutputDebugStringA(ex.what());
        std::cout << "LoadPlugin error" << std::endl;
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
        //std::cout << checkFileExistence(modelPath) << std::endl;
        //std::cout << checkFileExistence(modelPath.substr(0, modelPath.size() - 4) + ".bin") << std::endl;
        std::cout << modelPath << " " << checkFileExistence(modelPath)  << std::endl;
        std::cout << (modelPath.substr(0, modelPath.size() - 4) + ".bin") << " " << checkFileExistence(modelPath.substr(0, modelPath.size() - 4) + ".bin") << std::endl;
        network_reader.ReadNetwork(modelPath);
        network_reader.ReadWeights(modelPath.substr(0, modelPath.size() - 4) + ".bin");
        network_reader.getNetwork().setBatchSize(1);
        network = network_reader.getNetwork();
    }
    catch (const std::exception & ex)
    {
        //OutputDebugStringA(ex.what());
        std::cout << "ReadModel error" << std::endl;
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
        //OutputDebugStringA(ex.what());
        std::cout << "ConfigureInput error" << std::endl;
        ret = false;
    }

    return ret;
}

bool ConfigureOutput(CNNNetwork& network, OutputsDataMap& output_info, std::string& output_name, const Precision precision, const Layout layout)
{
    bool ret = true;
    
    try
    {
        std::cout << "ConfigureOutput stert" << std::endl;
        output_info = OutputsDataMap(network.getOutputsInfo());
        std::cout << "OutputsDataMap end" << std::endl;

        for (auto&& output : output_info)
        {
            output_name = output.first;
            output.second->setPrecision(precision);
            std::cout << "setPrecision end" << std::endl;
            //ここ死んでる
            output.second->setLayout(layout);
            std::cout << "setLayout end" << std::endl;
        }
    }
    catch (const std::exception & ex)
    {
        //OutputDebugStringA(ex.what());
        std::cout << "ConfigureOutput error" << std::endl;
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
        std::cout << "LoadModel end" << std::endl;
    }
    catch (const std::exception & ex)
    {
        //OutputDebugStringA(ex.what());
        std::cout << "LoadModel error" << std::endl;
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
        //OutputDebugStringA(ex.what());
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
        //OutputDebugStringA(ex.what());
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
        //OutputDebugStringA(ex.what());
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
        //OutputDebugStringA(ex.what());
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
    //std::string device = "GPU";
    std::string device = "MYRIAD";
    std::string modelPath = "/home/pi/pytorch-ssd/models/forasp/mbv3-ssd-cornv1.xml";
    int result = 0;


    //set up camera
    int camera_id = 0;
    double fps = 30.0;
    double width = 640.0;
    double height = 480.0;
    float threshold = 0.5;
    cv::VideoCapture cap(camera_id);
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
    if(ConfigureInput(network, input_info, input_name, Precision::U8, Layout::NCHW) == -1){
        std::cout << "ConfigureInput error!" << std::endl;
        return 0;
    }
    std::cout << "configure input end" << std::endl;
    if(ConfigureOutput(network, output_info, output_name, Precision::FP32, Layout::CHW) == -1){
        std::cout << "ConfigureOutput error!" << std::endl;
        return 0;
    }
    std::cout << "configure output end" << std::endl;
    if(LoadModel(network, plugin, executable_network) == -1){
        std::cout << "LoadModel error!" << std::endl;
        return 0;
    }
    std::cout << "LoadModel end" << std::endl;
    if(CreateInferRequest(executable_network, async_infer_request) == -1){
        std::cout << "CreateInferRequest error!" << std::endl;
        return 0;
    }
    std::cout << "CreateInferRequest end" << std::endl;

    //get output information
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
    }
    
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
        const float *output_concat = async_infer_request->GetBlob(output_name[0])->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
        const float *output_softmax = async_infer_request->GetBlob(output_name[0])->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
        
        std::vector<std::vector<int> > boxes;
        std::vector<std::vector<float> > labels;

        for(size_t i = 0; i < numDetections[1]; i++){
            //std::cout << detection_bound[i * objectSizes[0]] *300 << " " << detection_bound[i * objectSizes[0] + 1] *300<< " " << detection_bound[i * objectSizes[0] + 2] *300<< " " << detection_bound[i * objectSizes[0] + 3] *300<< std::endl;
            //std::cout << detection_soft[i * objectSizes[1]] << " " << detection_soft[i * objectSizes[1] + 1] << " " << detection_soft[i * objectSizes[1] + 2] << std::endl;
            //cv::rectangle(frame, cv::Point(static_cast<int>(detection_bound[i * objectSizes[0]] *300),static_cast<int>(detection_bound[i * objectSizes[0] + 1] *300)), cv::Point(static_cast<int>(detection_bound[i * objectSizes[0] + 2] *300),static_cast<int>(detection_bound[i * objectSizes[0] + 3] *300)), cv::Scalar(255,0,0), 2);
            std::vector<int> box;
            for(int l = 0; l < objectSizes[0]; l++){ //concat
                int mid = static_cast<int>(output_concat[i * objectSizes[0] + l] * 300);
                if(output_concat[i * objectSizes[0] + l] > 2 && (i * objectSizes[0] + l) < 9000)
                    std::cout << i * objectSizes[0] + l << " " <<  output_concat[i * objectSizes[0] + l]  << std::endl;
                box.push_back(mid);
            }
            boxes.push_back(box);
            std::vector<float> label;
            for(int l = 0; l < objectSizes[1]; l++){ //softmax
                float mid = output_softmax[i * objectSizes[1] + l];
                label.push_back(mid);
            }
            labels.push_back(label);            
        }
        
        for(size_t i = 0; i < numDetections[1]; i++){
            for(int l = 1; l < 2; l++){
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
                        //std::cout << xmin << " " << ymin << " " << xmax << " " << ymax << std::endl;
                    
                    cv::rectangle(frame, cv::Point(xmin,ymin), cv::Point(xmax,ymax), cv::Scalar(255,0,0), 2);
                }
            }
        }

        cv::imshow("frame", frame);
    }
}
