#include "stdafx.h"
#include <iostream>
#include <iomanip>
#include "opencv2/highgui.hpp"
#include "../VisionLibrary/VisionAPI.h"
#include "../VisionLibrary/StopWatch.h"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaarithm.hpp"

namespace AOI
{
namespace Vision
{

static const int REPEAT_TIME = 10;

static void TransposeSpeedTest_1()
{
    std::cout << std::endl << "--------------------------------------------------------";
    std::cout << std::endl << "TRANSPOSE WITHOUT PREPARED MEMORY SPEED TEST #1 STARTING";
    std::cout << std::endl << "--------------------------------------------------------";
    std::cout << std::endl;

    float totalTime = 0;
    
    for( int i = 0; i < REPEAT_TIME; ++ i ) {
        cv::Mat matImage = cv::Mat::eye(2048, 2040, CV_32FC1);

        CStopWatch stopWatch;
        cv::transpose(matImage, matImage);
        totalTime += stopWatch.Now();
    }
    float averageTime = totalTime / REPEAT_TIME;
    std::cout << "Repeat " << REPEAT_TIME << " times. Total time: " << totalTime << ", average time: " << averageTime << std::endl;
}

static void TransposeSpeedTest_2()
{
    std::cout << std::endl << "-----------------------------------------------------";
    std::cout << std::endl << "TRANSPOSE WITH PREPARED MEMORY SPEED TEST #1 STARTING";
    std::cout << std::endl << "-----------------------------------------------------";
    std::cout << std::endl;

    cv::Mat matImage = cv::Mat::eye(2048, 2040, CV_32FC1);
    cv::Mat matResult = cv::Mat::zeros(2040, 2048, CV_32FC1);
    CStopWatch stopWatch;
    for( int i = 0; i < REPEAT_TIME; ++ i ) {
        cv::transpose(matImage, matResult);
    }

    float totalTime = ToFloat(stopWatch.Now());
    auto averageTime = totalTime / REPEAT_TIME;
    std::cout << "Repeat " << REPEAT_TIME << " times. Total time: " << totalTime << ", average time: " << averageTime << std::endl;
}

static void TransposeSpeedTest_3()
{
    std::cout << std::endl << "-------------------------------------------------------------";
    std::cout << std::endl << "CUDA TRANSPOSE WITHOUT PREPARED MEMORY SPEED TEST #1 STARTING";
    std::cout << std::endl << "-------------------------------------------------------------";
    std::cout << std::endl;

    float totalTime = 0;
    
    for( int i = 0; i < REPEAT_TIME; ++ i ) {
        cv::cuda::GpuMat matImage(2048, 2040, CV_32FC1);

        CStopWatch stopWatch;
        cv::cuda::transpose(matImage, matImage);
        totalTime += stopWatch.Now();
    }
    auto averageTime = totalTime / REPEAT_TIME;
    std::cout << "Repeat " << REPEAT_TIME << " times. Total time: " << totalTime << ", average time: " << averageTime << std::endl;
}

static void TransposeSpeedTest_4()
{
    std::cout << std::endl << "----------------------------------------------------------";
    std::cout << std::endl << "CUDA TRANSPOSE WITH PREPARED MEMORY SPEED TEST #1 STARTING";
    std::cout << std::endl << "----------------------------------------------------------";
    std::cout << std::endl;

    cv::cuda::GpuMat matImage(2048, 2040, CV_32FC1);
    cv::cuda::GpuMat matResult(2040, 2048, CV_32FC1);
    CStopWatch stopWatch;
    for( int i = 0; i < REPEAT_TIME; ++ i ) {
        cv::cuda::transpose(matImage, matResult);
    }

    float totalTime = ToFloat(stopWatch.Now());
    auto averageTime = totalTime / REPEAT_TIME;
    std::cout << "Repeat " << REPEAT_TIME << " times. Total time: " << totalTime << ", average time: " << averageTime << std::endl;
}

void TestTransposeSpeed() {
    TransposeSpeedTest_1();
    TransposeSpeedTest_2();
    TransposeSpeedTest_3();
    TransposeSpeedTest_4();
}

static void TestRotateImage90SpeedTest(int caseNo, int type, const std::string& testName) {
    std::cout << std::endl << "-----------------------------------------------------";
    std::cout << std::endl << "ROTATE IMAGE 90 Degree of " << testName << " SPEED TEST #" << caseNo << " STARTING";
    std::cout << std::endl << "-----------------------------------------------------";
    std::cout << std::endl;

    float totalTime = 0;
    cv::Mat matResult = cv::Mat::zeros(2040, 2048, type);
    for (int i = 0; i < REPEAT_TIME; ++i) {
        cv::Mat matImage = cv::Mat::eye(2048, 2040, type);

        CStopWatch stopWatch;
        cv::transpose(matImage, matResult);
        cv::flip(matResult, matResult, 1); //transpose+flip(1)=CW

        totalTime += stopWatch.Now();
    }

    float averageTime = totalTime / REPEAT_TIME;
    std::cout << "Repeat " << REPEAT_TIME << " times. Total time: " << totalTime << ", average time: " << averageTime << std::endl;
}

void TestRotateSpeed() {
    TestRotateImage90SpeedTest(0, CV_8UC1, "CV_8UC1");
    TestRotateImage90SpeedTest(1, CV_32FC1, "CV_32FC1");
}

}
}