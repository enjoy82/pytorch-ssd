//参考元 https://qiita.com/fan2tamo/items/18f418bd6d23686621ea

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include <map>
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

#include "object_detecting.hpp"


InferenceEngine::Core core;
InferenceEngine::CNNNetwork network;
InferenceEngine::ExecutableNetwork executableNetwork;
InferenceEngine::InferRequest inferRequest;
std::string inputName;
//今回構築したモデルが2つoutputを返すのでvector
std::vector<std::string> outputName;

namespace
{

    /*
    //nms関連の関数はここに退避しておく、(ファイル大きいけど大丈夫か？)
    //何返す？
    bool areaOf(leftTop, rightBottom)
    {

    }
    //何返す？
    bool iouOf(leftTop, rightBottom, double eps = 1e-5)
    {

    }
     */
    std::vector<std::vector<float> > hardNms(std::vector<std::vector<float> > &boxScores, int candidate_size=200)
    {
        //各ラベルでのバウンディングボックスの被りを消す
        std::vector<std::vector<float> > res = boxScores;
        return res;
        /*
        std::vector<std::vector<float> >　picked;
        //降順ソート
        std::sort(boxScores.begin(), boxScores.end(), [](auto &x, auto &y){x[4] > y[4];});
        while()
        */
    }

}

ObjectDetector::ObjectDetector()
{
}

void ObjectDetector::matU8ToBlobFP32(const cv::Mat &orgImage, InferenceEngine::Blob::Ptr &blob)
{
    //NHWC -> NCHW and U8 -> FP32の変換をし、blobに入れる関数
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

    float *blobData = mblobHolder.as<float *>();

    cv::Mat resizedImg(orgImage);
    if (static_cast<int>(width) != orgImage.size().width ||
            static_cast<int>(height) != orgImage.size().height) {
        cv::resize(orgImage, resizedImg, cv::Size(width, height));
    }

    //NHWC -> NCHW and U8 -> FP32
    int imageSize = width * height;
    for (size_t pid = 0; pid < imageSize; pid++) {
        // Iterate over all channels 
        for (size_t ch = 0; ch < 3; ++ch) {
            //-1 ~ 1 で正規化する
            blobData[ch * imageSize + pid] = ((float)resizedImg.data[pid*3 + ch] - 127) / 128.0;
        }
    }
}


bool ObjectDetector::readNetwork(const std::string &modelPath)
{
    bool ret = true;
    try
    { 
        // IR形式のみ読み込み可能
        std::string binPath = modelPath.substr(0, modelPath.length() - 4) + ".bin";
        network = core.ReadNetwork(modelPath, binPath);
    }
    catch (...)
    {
        printf("Error ReadNetwork() \r\n");
        ret = false;
    }

    return ret;
}

bool ObjectDetector::setInputInfos()
{
    bool ret = true;
    try
    {   
        InferenceEngine::InputsDataMap inputsInfo(network.getInputsInfo());
        for (auto & item : inputsInfo) {
            /** Working with first input tensor that stores image **/
            if (item.second->getInputData()->getTensorDesc().getDims().size() == 4) {
                inputName = item.first;
                //inputInfo = item.second;
                /** Creating first input blob **/
                InferenceEngine::Precision inputPrecision = InferenceEngine::Precision::FP32;
                item.second->setPrecision(inputPrecision);
                item.second->setLayout(InferenceEngine::Layout::NCHW);
            }
        }
    }
    catch(...)
    {
        printf("Error setInputInfos() \r\n");
        ret = false;
    }

    return ret;
}

bool ObjectDetector::setOutputInfos()
{
    bool ret = true;
    try
    {
        InferenceEngine::OutputsDataMap outputsInfo(network.getOutputsInfo());

        for (const auto& out : outputsInfo) {
            outputName.push_back(out.first);
            InferenceEngine::DataPtr outputInfo = out.second;
            outputInfo->setPrecision(InferenceEngine::Precision::FP32);
            outputInfo->setLayout(InferenceEngine::Layout::CHW);
            //outputInfos.push_back(outputInfo);
            std::cout << std::endl;
        }
    }
    catch(...)
    {
        printf("Error setOutputInfos() \r\n");
        ret = false;
    }

    return ret;
}

bool ObjectDetector::loadNetwork(std::string &device)
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

bool ObjectDetector::createInferRequest()
{
    bool ret = true;
    try
    {
        inferRequest = executableNetwork.CreateInferRequest();
    }
    catch (...)
    {
        ret = false;
        printf("Error : CreateInferRequest()\r\n");
    }

    return ret;
}

bool ObjectDetector::setInputData(cv::Mat &image){
    //入力を識別機に突っ込む関数
    bool ret = true;
    
    try
    {
        //ここでblobをとってくる
        InferenceEngine::Blob::Ptr inputBlob = inferRequest.GetBlob(inputName);
        //matをblobの中に突っ込む
        matU8ToBlobFP32(image, inputBlob);
    }
    catch (...)
    {
        ret = false;
        printf("Error : SetInputData()\r\n");
    }

    return ret;
}

bool ObjectDetector::infer()
{
    bool ret = true;

    try
    {
        inferRequest.Infer();
        //いる？
        inferRequest.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
    }
    catch(...)
    {
        ret = false;
        printf("Error : infer()\r\n");
    }
    
    return ret;
}

//何返そうかのう & エラーハンドリングも考える必要あり、一番ここバグる可能性高い
std::vector<std::vector<std::vector<float> > > ObjectDetector::getOutputData(double threshold)
{
    const float *outputConcat = (inferRequest.GetBlob(outputName[0]))->cbuffer().as<const InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type *>();
    const float *outputSoftmax = (inferRequest.GetBlob(outputName[1]))->cbuffer().as<const InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type *>();
    //あとで扱いやすいようにvectorに置き換えておく label順に{xmin, ymin, xmax, ymax, score}でvectorを持つ

    //要素数と、サイズをハードコーディングしてるので、後々直したい
    int LABEL = 3;
    float XMIN = 0.0, YMIN = 0.0, XMAX = 300.0, YMAX = 300.0;
    std::vector<std::vector<std::vector<float> > > boxScores(LABEL);

    for(size_t i = 0; i < 3000; i++)
    {    
        float xmin = std::max(XMIN, outputConcat[i * 4] * 300);
        float ymin = std::max(YMIN, outputConcat[i * 4 + 1] * 300);
        float xmax = std::min(XMAX, outputConcat[i * 4 + 2] * 300);
        float ymax = std::min(YMAX, outputConcat[i * 4 + 3] * 300);

        bool pushFlag = false;
        std::vector<float> label;
        for(int l = 0; l < LABEL; l++){ //softmax
            float score = outputSoftmax[i * LABEL + l];
            if(l != 0 && score > threshold){
                std::vector<float> boxScore = {xmin, ymin, xmax, ymax, score};
                boxScores[l].push_back(boxScore);
            }

        }
    }
    
    std::vector<std::vector<std::vector<float> > > pickedBoxScores(LABEL);
    for(int i = 1; i < LABEL; i++)
    {
        if(boxScores[i].size() == 0) continue;
        std::vector<std::vector<float> > boxScoresNms = hardNms(boxScores[i]);
        pickedBoxScores[i] = boxScoresNms;
    }
    return pickedBoxScores;
}

//モデルの構築から推論までのセットアップを行う
bool ObjectDetector::initObjectDetector(std::string &modelPath, std::string &device)
{
    
    if(!readNetwork(modelPath))
    {
        std::cout << "ReadNetwork error!" << std::endl;
        return false;
    }
    std::cout << "ReadNetwork end" << std::endl;
    if(!setInputInfos())
    {
        std::cout << "setInputInfos error!" << std::endl;
        return false;
    }
    std::cout << "setInputInfos end" << std::endl;
    if(!setOutputInfos())
    {
        std::cout << "setOutputInfos error!" << std::endl;
        return false;
    }
    std::cout << "setOutputInfos end" << std::endl;
    if(!loadNetwork(device))
    {
        std::cout << "LoadNetwork error!" << std::endl;
        return false;
    }
    std::cout << "loadNetwork end" << std::endl;
    if(!createInferRequest())
    {
        std::cout << "CreateInferRequest error!" << std::endl;
        return false;
    }

    std::cout << "finished initialize" << std::endl;
    return true;
}