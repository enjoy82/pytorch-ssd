#include <iostream>
#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>
#include <ie_iextension.h>
#include<fstream>
//#include <ext_list.hpp>
#include <string>

class object_detector{
    private:
        InferenceEngine::Core core;
        std::string input_name;
        std::vector<std::string> output_names; //first is concat, second is softmax
        std::vector<int> numDetections;
        std::vector<int> objectSizes;
        float threshold;

        void init_object_detector();
    public:
        //コンストラクタ
        object_detector(std::string &modelPath, std::string &device, float threshold);
        //デストラクタ
        ~object_detector();
        init_object_detect();
}