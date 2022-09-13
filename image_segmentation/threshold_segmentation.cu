#include "iostream"
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/imgcodecs.hpp"

int main (int argc, char* argv[])
{
    try
    {
        cv::Mat src_host = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);
        cv::cuda::GpuMat dst, src;
        src.upload(src_host);

        cv::cuda::threshold(src, dst, 100.0, 200.0, cv::THRESH_BINARY);

        cv::Mat result_host;
        dst.download(result_host);

        cv::imshow("Result", result_host);
        cv::waitKey();
    }
    catch(const cv::Exception& ex) { std::cout << "Exception error: " << ex.what() << std::endl; }
    return 0;
}
/* compile: nvcc -I /usr/local/include/opencv4/ -lopencv_core -lopencv_highgui -lopencv_cudaarithm -lopencv_imgproc -lopencv_imgcodecs threshold_segmentation.cu -o threshold_segmentation
*/