// Daniel Bandala @ dec-2022
#include <iostream>
#include <ctime>
#include <cmath>
#include "bits/time.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

#define TestCUDA true

int main() {
    std::clock_t begin = std::clock();

        try {
            cv::String filename = "./2022-11-06-143613.jpg"; //change file name properly
            cv::Mat resultHost;
            cv::Mat srcHost = cv::imread(filename, cv::IMREAD_GRAYSCALE);

            for(int i=0; i<1000; i++) {
                if(TestCUDA) {
                    cv::cuda::GpuMat dst, src;
                    src.upload(srcHost);

                    cv::cuda::threshold(src,dst,128.0,255.0,cv::THRESH_BINARY);
                    cv::cuda::bilateralFilter(dst,dst,3,1,1);
                    dst.download(resultHost);
                } else {
                    cv::Mat dst;
                    cv::bilateralFilter(srcHost,dst,3,1,1);
                }
            }

            cv::imshow("Result",resultHost);
            cv::waitKey();

        } catch(const cv::Exception& ex) {
            std::cout << "Error: " << ex.what() << std::endl;
        }

    std::clock_t end = std::clock();
    std::cout << double(end-begin) / CLOCKS_PER_SEC  << std::endl;
}

// compile: nvcc test.cpp `pkg-config opencv --cflags --libs` -o test