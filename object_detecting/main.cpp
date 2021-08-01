#include <opencv2/opencv.hpp>
#include <iostream>

#include "object_detecting.hpp"

int main(){
    std::cout << "Start infer" << std::endl;
    std::string imagename = "/home/pi/samples/build/hikage_010_can.JPG";
    std::string model_path = "/home/pi/pytorch-ssd/object_detecting/mbv3-ssd-cornv1.xml";
    std::string device = "MYRIAD";

    cv::Mat image = cv::imread(imagename);
    std::cout << "read end" << std::endl;
    ObjectDetector objectDetector;
    std::cout << "inti start" << std::endl;
    objectDetector.initObjectDetector(model_path, device);
    std::cout << "setInputData start" << std::endl;
    objectDetector.setInputData(image);
    std::cout << "infer start" << std::endl;
    objectDetector.infer();
    std::cout << "getOutputData start" << std::endl;
    std::vector<std::vector<std::vector<float> > > pickedBox = objectDetector.getOutputData(0.3);

    cv::resize(image, image, cv::Size(300, 300));

    for(size_t i = 1; i < pickedBox.size(); i++){
        for(int l = 0; l < pickedBox[i].size(); l++){
            int xmin = std::max(0, static_cast<int>(pickedBox[i][l][0]));
            int ymin = std::max(0, static_cast<int>(pickedBox[i][l][1]));
            int xmax = std::min(300, static_cast<int>(pickedBox[i][l][2]));
            int ymax = std::min(300, static_cast<int>(pickedBox[i][l][3]));
            
            //if(xmin > 300 || ymin > 300 || xmax < 0 || ymax < 0)
            std::cout << l << std::endl;
            std::cout << xmin << " " << ymin << " " << xmax << " " << ymax << " " << pickedBox[i][l][4] << std::endl;
            
            cv::rectangle(image, cv::Point(xmin,ymin), cv::Point(xmax,ymax), cv::Scalar(255,0,0), 2);

        }
    }
    cv::imwrite("/home/pi/pytorch-ssd/object_detecting/result.png", image);
}