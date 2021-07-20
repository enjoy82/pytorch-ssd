//from https://qiita.com/fan2tamo/items/36bc8f9657d1a430aa54#8-process-output
#include <iostream>
#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>
#include <ie_iextension.h>
#include<fstream>
#include <algorithm>
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
        OutputDebugStringA(ex.what());
        std::cout << "LoadPlugin error" << std::endl;
        ret = false;
    }

    return ret;
}


bool ReadModel(const std::string &modelPath, CNNNetReader& network_reader)
{
    bool ret = true;

    try
    {
        //std::cout << checkFileExistence(modelPath) << std::endl;
        //std::cout << checkFileExistence(modelPath.substr(0, modelPath.size() - 4) + ".bin") << std::endl;
        std::cout << modelPath << " " << checkFileExistence(modelPath)  << std::endl;
        std::cout << (modelPath.substr(0, modelPath.size() - 4) + ".bin") << " " << checkFileExistence(modelPath.substr(0, modelPath.size() - 4) + ".bin") << std::endl;
        network_reader.ReadNetwork(modelPath);
        network_reader.ReadWeights(modelPath.substr(0, modelPath.size() - 4) + ".bin");
    }
    catch (const std::exception & ex)
    {
        OutputDebugStringA(ex.what());
        std::cout << "ReadModel error" << std::endl;
        ret = false;
    }

    return ret;
}
/*
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
        std::cout << "ConfigureInput error" << std::endl;
        ret = false;
    }

    return ret;
}
*/
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
        OutputDebugStringA(ex.what());
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
/*
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
*/
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

//やばい
template <typename T>
bool PrepareInput(InferenceEngine::InferRequest & infer_request, T & input_info, const cv::Mat & image)
{
    bool ret = true;

    try
    {
        
        for (auto & item : input_info) {
            auto input_data = item.second;
            Blob::Ptr imgBlob = wrapMat2Blob(image);
            infer_request.SetBlob(item.first, imgBlob);
        }
        
    }
    catch (const std::exception & ex)
    {
        OutputDebugStringA(ex.what());
        std::cout << "PrepareInput error!" << std::endl;
        ret = false;
    }

    return ret;
}


bool Infer(InferenceEngine::InferRequest & infer_request)
{
    bool ret = true;

    try
    {
        infer_request.StartAsync();
        infer_request.Wait(IInferRequest::WaitMode::RESULT_READY);
    }
    catch (const std::exception & ex)
    {
        OutputDebugStringA(ex.what());
        std::cout << "Infer error!" << std::endl;
        ret = false;
    }

    return ret;
}

//TODO 書き換え
/*
int ProcessOutput(InferenceEngine::InferRequest & infer_request, const std::string& output_name)
{

    int result = 0;
    float buf= 0;

    try
    {
        const float* oneHotVector = infer_request->GetBlob(output_name)->buffer().as<float*>();

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
*/

class object_detector{
    private:
    public:
        //コンストラクタ
        object_detector(std::string modelPath){

        }
        //デストラクタ
        ~object_detector(){

        }
}

int main(){
    std::string modelPath = "C:\\Users\\Naoya Yatsu\\Desktop\\code\\pytorch-ssd\\live_demo_cpp\\models\\mbv3-ssd-cornv1.xml";
    std::string device = "CPU";
    std::vector<std::string> output_names; //first is concat, second is softmax
    std::vector<int> numDetections;
    std::vector<int> objectSizes;
    int camera_id = 0;
    double fps = 30.0;
    double width = 640.0;
    double height = 480.0;
    double input_width = 300.0;
    double input_height = 300.0;
}

//TODO refactor
int main(){
    InferenceEngine::Core core;
    CNNNetReader network_reader;
    std::string input_name;
    //std::string output_name;
    //std::string device = "GPU";
    //std::string device = "MYRIAD";
    std::string device = "CPU";
    std::string modelPath = "C:\\Users\\Naoya Yatsu\\Desktop\\code\\pytorch-ssd\\live_demo_cpp\\models\\mbv3-ssd-cornv1.xml";
    int result = 0;


    //set up camera
    int camera_id = 0;
    double fps = 30.0;
    double width = 640.0;
    double height = 480.0;
    double input_width = 300.0;
    double input_height = 300.0;
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
    //LoadPlugin(device, plugin);
    ReadModel(modelPath, network_reader);
    auto network = network_reader.getNetwork();
    InferenceEngine::InputsDataMap input_info(network.getInputsInfo());
    InferenceEngine::OutputsDataMap output_info(network.getOutputsInfo());
    //ConfigureInput(network, input_info, input_name, Precision::U8, Layout::NCHW);
    std::cout << "configure input end" << std::endl;

    for (auto &item : input_info) {
        input_name = item.first;
        auto input_data = item.second;
        input_data->setPrecision(Precision::U8);
        input_data->setLayout(Layout::NCHW);
        input_data->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
        input_data->getPreProcess().setColorFormat(ColorFormat::RGB);
    }
    std::cout << typeid(input_info).name() << std::endl;
    std::cout << "input_data input end" << std::endl;
    std::vector<std::string> output_names; //first is concat, second is softmax
    std::vector<int> numDetections;
    std::vector<int> objectSizes;
    for (auto &item : output_info) {
        output_names.push_back(item.first);
        auto output_data = item.second;
        std::cout << "output item now" << std::endl;
        output_data->setPrecision(Precision::FP32);
        std::cout << "output item now2" << std::endl;
        output_data->setLayout(Layout::CHW);
        std::cout << "output item now3" << std::endl;
        const SizeVector outputDims = output_data->getTensorDesc().getDims();
        std::cout << "output item now4" << std::endl;
        numDetections.push_back(outputDims[1]);
        std::cout << "output item now5" << outputDims[2]  << " " << outputDims[1]  << " " << outputDims[0]<< std::endl;
        objectSizes.push_back(outputDims[2]);
    }
    for(int i = 0; i < output_names.size(); i++){
        std::cout << output_names[i] << " " << numDetections[i] << " " << objectSizes[i] << std::endl;
    }
    //ConfigureOutput(network, output_info, output_name, Precision::FP32, Layout::NC);
    std::cout << "configure output end" << std::endl;
    //TODO 変更
    std::map<std::string, std::string> config = {{ PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES }};
    auto executable_network = core.LoadNetwork(network, device, config);
    
    //LoadModel(network, plugin, executable_network);
    std::cout << "LoadModel end" << std::endl;

    std::chrono::milliseconds timespan(3000); // or whatever

    //CreateInferRequest(executable_network, async_infer_request);
    InferenceEngine::InferRequest infer_request = executable_network.CreateInferRequest();
    std::cout << typeid(infer_request).name() << std::endl;
    std::cout << "CreateInferRequest end" << std::endl;
    /*
    DataPtr& output = output_info.begin()->second;
    const SizeVector outputDims = output->getTensorDesc().getDims();
    const int numDetections = outputDims[2];
    const int objectSize = outputDims[3];
    */
    //std::this_thread::sleep_for(timespan);
    while(cap.read(frame)){
        //cv::imshow("frame", frame);
        cv::resize(frame, frame, cv::Size(), input_width/frame.cols, input_height/frame.rows);
        //std::cout << frame.rows << " " << frame.cols << std::endl;
        int key = cv::waitKey(1);
		if(key == 'q'){
			cv::destroyWindow("frame");
			break;
        }
        //std::cout << "call" << std::endl;
        PrepareInput(infer_request, input_info, frame);
        Infer(infer_request);
        //result = ProcessOutput(async_infer_request, output_name);
        //cv::putText(frame, "Test Frame", cv::Point(50,50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,200), 2, false);
        const float *output_concat = infer_request.GetBlob(output_names[0])->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
        const float *output_softmax = infer_request.GetBlob(output_names[1])->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
        
        std::vector<int> label, xmin, xmax, ymin, ymax;
        for(int i = 0; i < numDetections[1]; i++){
            for(int l = 1; l < 2; l++){ //first is background
                if(output_softmax[i * objectSizes[1] + l] > threshold){
                    label.push_back(l);
                    if(l == 2){
                        cv::putText(frame, "can Frame", cv::Point(50,50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,200), 2, false);
                    }else{
                        std::cout << i * objectSizes[1] + l << std::endl;
                        cv::putText(frame, "cone Frame", cv::Point(50,50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,200), 2, false);
                    }
                    xmin.push_back(std::max(0, static_cast<int>(output_concat[i * objectSizes[0]] * input_width)));
                    ymin.push_back(std::max(0, static_cast<int>(output_concat[i * objectSizes[0] + 2] * input_width)));
                    xmax.push_back(std::min(static_cast<int>(input_width), static_cast<int>(output_concat[i * objectSizes[0] + 1] * input_width)));
                    ymax.push_back(std::min(static_cast<int>(input_height), static_cast<int>(output_concat[i * objectSizes[0] + 3] * input_height)));
                    std::cout << output_softmax[i * objectSizes[1] + l] << " " <<  xmin[xmin.size() - 1] << " " << ymin[ymin.size() - 1] << " " << xmax[xmax.size() - 1] << " " << ymax[ymax.size() - 1]  << std::endl;
                    cv::rectangle(frame, cv::Point(xmin[xmin.size() - 1],ymin[ymin.size() - 1]), cv::Point(xmax[xmax.size() - 1],ymax[ymax.size() - 1]), cv::Scalar(255,0,0), 2);
                }
            }
        }
        
        
        cv::imshow("frame", frame);
    }
}
