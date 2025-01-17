﻿#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "../VisionLibrary/VisionAPI.h"
#include <iostream>
#include <iomanip>

namespace AOI
{
namespace Vision
{

static void PrintMatchTmplRpy(const PR_MATCH_TEMPLATE_RPY &stRpy) {
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Match template status " << ToInt32(stRpy.enStatus) << std::endl;
    std::cout << "Match template score " << stRpy.fMatchScore << std::endl;
    if ( VisionStatus::OK == stRpy.enStatus ) {
        std::cout << "Match template result " << stRpy.ptObjPos.x << ", " << stRpy.ptObjPos.y << std::endl;
        std::cout << "Match template angle " << stRpy.fRotation << std::endl;
    }
}

void TestTmplMatch_1()
{
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl << "MATCH TEMPLATE REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl;

    PR_LRN_TEMPLATE_CMD stLrnCmd;
    PR_LRN_TEMPLATE_RPY stLrnRpy;
    stLrnCmd.matInputImg = cv::imread("./data/Template.png", cv::IMREAD_GRAYSCALE);
    stLrnCmd.rectROI = cv::Rect (80, 80, 100, 100 );
    stLrnCmd.enAlgorithm = PR_MATCH_TMPL_ALGORITHM::SQUARE_DIFF_NORMED;
    PR_LrnTmpl ( &stLrnCmd, &stLrnRpy );
    if ( stLrnRpy.enStatus != VisionStatus::OK ) {
        std::cout << "Failed to learn template." << std::endl;
        return;
    }

    PR_MATCH_TEMPLATE_CMD stCmd;
    PR_MATCH_TEMPLATE_RPY stRpy;
    stCmd.matInputImg = cv::imread("./data/FindAngle_1.png", cv::IMREAD_GRAYSCALE);
    stCmd.rectSrchWindow = cv::Rect(0, 0, stCmd.matInputImg.cols, stCmd.matInputImg.rows );
    stCmd.enAlgorithm = PR_MATCH_TMPL_ALGORITHM::SQUARE_DIFF_NORMED;
    stCmd.nRecordId = stLrnRpy.nRecordId;
    stCmd.bSubPixelRefine = true;
    stCmd.enMotion = PR_OBJECT_MOTION::EUCLIDEAN;
    
    PR_MatchTmpl(&stCmd, &stRpy);
    PrintMatchTmplRpy ( stRpy );
}

void TestTmplMatch_2()
{
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl << "MATCH TEMPLATE REGRESSION TEST #2 STARTING";
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl;

    PR_LRN_TEMPLATE_CMD stLrnCmd;
    PR_LRN_TEMPLATE_RPY stLrnRpy;
    stLrnCmd.matInputImg = cv::imread("./data/CapsLock_1.png");
    stLrnCmd.rectROI = cv::Rect (0, 0, stLrnCmd.matInputImg.cols, stLrnCmd.matInputImg.rows );
    stLrnCmd.enAlgorithm = PR_MATCH_TMPL_ALGORITHM::HIERARCHICAL_EDGE;
    PR_LrnTmpl ( &stLrnCmd, &stLrnRpy );
    if ( stLrnRpy.enStatus != VisionStatus::OK ) {
        std::cout << "Failed to learn template." << std::endl;
        return;
    }

    PR_MATCH_TEMPLATE_CMD stCmd;
    PR_MATCH_TEMPLATE_RPY stRpy;
    stCmd.matInputImg = cv::imread("./data/CapsLock_2.png", cv::IMREAD_GRAYSCALE);
    stCmd.rectSrchWindow = cv::Rect(0, 0, stCmd.matInputImg.cols, stCmd.matInputImg.rows );
    stCmd.enAlgorithm = PR_MATCH_TMPL_ALGORITHM::HIERARCHICAL_EDGE;
    stCmd.nRecordId = stLrnRpy.nRecordId;
    stCmd.enMotion = PR_OBJECT_MOTION::TRANSLATION;
    
    PR_MatchTmpl(&stCmd, &stRpy);
    PrintMatchTmplRpy ( stRpy );
}

void TestTmplMatch_3()
{
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl << "MATCH TEMPLATE REGRESSION TEST #3 STARTING";
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl;

    PR_LRN_TEMPLATE_CMD stLrnCmd;
    PR_LRN_TEMPLATE_RPY stLrnRpy;
    stLrnCmd.matInputImg = cv::imread("./data/CapsLock_1.png");
    stLrnCmd.rectROI = cv::Rect (0, 0, stLrnCmd.matInputImg.cols, stLrnCmd.matInputImg.rows );
    stLrnCmd.enAlgorithm = PR_MATCH_TMPL_ALGORITHM::SQUARE_DIFF_NORMED;
    PR_LrnTmpl ( &stLrnCmd, &stLrnRpy );
    if ( stLrnRpy.enStatus != VisionStatus::OK ) {
        std::cout << "Failed to learn template." << std::endl;
        return;
    }

    PR_MATCH_TEMPLATE_CMD stCmd;
    PR_MATCH_TEMPLATE_RPY stRpy;
    stCmd.matInputImg = cv::imread("./data/CapsLock_2.png", cv::IMREAD_GRAYSCALE);
    stCmd.rectSrchWindow = cv::Rect(0, 0, stCmd.matInputImg.cols, stCmd.matInputImg.rows );
    stCmd.enAlgorithm = PR_MATCH_TMPL_ALGORITHM::SQUARE_DIFF_NORMED;
    stCmd.nRecordId = stLrnRpy.nRecordId;
    stCmd.enMotion = PR_OBJECT_MOTION::TRANSLATION;
    
    PR_MatchTmpl(&stCmd, &stRpy);
    PrintMatchTmplRpy ( stRpy );
}

void TestTmplMatch_4()
{
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl << "MATCH TEMPLATE REGRESSION TEST #4 STARTING";
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl;

    PR_LRN_TEMPLATE_CMD stLrnCmd;
    PR_LRN_TEMPLATE_RPY stLrnRpy;
    stLrnCmd.matInputImg = cv::imread("./data/MtfTargetTmpl.png");
    stLrnCmd.rectROI = cv::Rect (0, 0, stLrnCmd.matInputImg.cols, stLrnCmd.matInputImg.rows );
    stLrnCmd.enAlgorithm = PR_MATCH_TMPL_ALGORITHM::HIERARCHICAL_AREA;
    PR_LrnTmpl ( &stLrnCmd, &stLrnRpy );
    if ( stLrnRpy.enStatus != VisionStatus::OK ) {
        std::cout << "Failed to learn template." << std::endl;
        return;
    }

    PR_MATCH_TEMPLATE_CMD stCmd;
    PR_MATCH_TEMPLATE_RPY stRpy;
    stCmd.matInputImg = cv::imread("./data/MtfTarget.png", cv::IMREAD_GRAYSCALE);
    stCmd.rectSrchWindow = cv::Rect(0, 0, stCmd.matInputImg.cols, stCmd.matInputImg.rows );
    stCmd.enAlgorithm = PR_MATCH_TMPL_ALGORITHM::HIERARCHICAL_AREA;
    stCmd.nRecordId = stLrnRpy.nRecordId;
    stCmd.enMotion = PR_OBJECT_MOTION::TRANSLATION;
    
    PR_MatchTmpl(&stCmd, &stRpy);
    PrintMatchTmplRpy ( stRpy );
}

void TestTmplMatch_5()
{
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl << "MATCH TEMPLATE REGRESSION TEST #5 STARTING";
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl;

    PR_LRN_TEMPLATE_CMD stLrnCmd;
    PR_LRN_TEMPLATE_RPY stLrnRpy;
    stLrnCmd.matInputImg = cv::imread("./data/MtfTargetTmpl.png");
    stLrnCmd.rectROI = cv::Rect(0, 0, stLrnCmd.matInputImg.cols, stLrnCmd.matInputImg.rows);
    stLrnCmd.enAlgorithm = PR_MATCH_TMPL_ALGORITHM::HIERARCHICAL_AREA;
    PR_LrnTmpl(&stLrnCmd, &stLrnRpy);
    if (stLrnRpy.enStatus != VisionStatus::OK) {
        std::cout << "Failed to learn template." << std::endl;
        return;
    }

    PR_MATCH_TEMPLATE_CMD stCmd;
    PR_MATCH_TEMPLATE_RPY stRpy;
    stCmd.matInputImg = cv::imread("./data/PCB_With_CAE.png", cv::IMREAD_GRAYSCALE);
    stCmd.rectSrchWindow = cv::Rect(0, 0, stCmd.matInputImg.cols, stCmd.matInputImg.rows );
    stCmd.enAlgorithm = PR_MATCH_TMPL_ALGORITHM::HIERARCHICAL_AREA;
    stCmd.nRecordId = stLrnRpy.nRecordId;
    stCmd.enMotion = PR_OBJECT_MOTION::TRANSLATION;
    
    PR_MatchTmpl(&stCmd, &stRpy);
    PrintMatchTmplRpy(stRpy);
}

void TestTmplMatch_6() {
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl << "MATCH TEMPLATE REGRESSION TEST #6 STARTING";
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl;

    PR_LRN_TEMPLATE_CMD stLrnCmd;
    PR_LRN_TEMPLATE_RPY stLrnRpy;

    stLrnCmd.matInputImg = cv::imread("./data/TestOCV.png");
    stLrnCmd.enAlgorithm = PR_MATCH_TMPL_ALGORITHM::SQUARE_DIFF_NORMED;
    stLrnCmd.rectROI = cv::Rect(311, 818, 180, 61);
    cv::Mat matMask = cv::Mat::ones(stLrnCmd.rectROI.size(), CV_8UC1) * 255;
    cv::rectangle(matMask, cv::Rect(10, 10, 160, 30), cv::Scalar(0), CV_FILLED);
    stLrnCmd.matMask = matMask;

    PR_LrnTmpl(&stLrnCmd, &stLrnRpy);
    if (stLrnRpy.enStatus != VisionStatus::OK) {
        std::cout << "Failed to learn template." << std::endl;
        return;
    }

    PR_MATCH_TEMPLATE_CMD stSrchCmd;
    PR_MATCH_TEMPLATE_RPY stSrchRpy;
    stSrchCmd.matInputImg = cv::imread("./data/TestOCV.png");
    stSrchCmd.enAlgorithm = PR_MATCH_TMPL_ALGORITHM::SQUARE_DIFF_NORMED;
    stSrchCmd.enMotion = PR_OBJECT_MOTION::TRANSLATION;
    stSrchCmd.rectSrchWindow = cv::Rect(223, 1035, 339, 216);
    stSrchCmd.nRecordId = stLrnRpy.nRecordId;

    PR_MatchTmpl(&stSrchCmd, &stSrchRpy);
    PrintMatchTmplRpy(stSrchRpy);
}

void TestTmplMatch_7() {
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl << "MATCH TEMPLATE REGRESSION TEST #7 STARTING";
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl;

    PR_LRN_TEMPLATE_CMD stLrnCmd;
    PR_LRN_TEMPLATE_RPY stLrnRpy;

    stLrnCmd.matInputImg = cv::imread("./data/TestSrchTmpl_7/Tmpl.png");
    stLrnCmd.enAlgorithm = PR_MATCH_TMPL_ALGORITHM::SQUARE_DIFF_NORMED;
    stLrnCmd.rectROI = cv::Rect(0, 0, stLrnCmd.matInputImg.cols, stLrnCmd.matInputImg.rows);

    PR_LrnTmpl(&stLrnCmd, &stLrnRpy);
    if (stLrnRpy.enStatus != VisionStatus::OK) {
        std::cout << "Failed to learn template." << std::endl;
        return;
    }

    PR_MATCH_TEMPLATE_CMD stSrchCmd;
    PR_MATCH_TEMPLATE_RPY stSrchRpy;
    stSrchCmd.matInputImg = cv::imread("./data/TestSrchTmpl_7/image.png");
    stSrchCmd.enAlgorithm = PR_MATCH_TMPL_ALGORITHM::SQUARE_DIFF_NORMED;
    stSrchCmd.enMotion = PR_OBJECT_MOTION::TRANSLATION;
    stSrchCmd.rectSrchWindow = cv::Rect(cv::Point(698, 674), cv::Point(2094, 2022));
    stSrchCmd.nRecordId = stLrnRpy.nRecordId;

    PR_MatchTmpl(&stSrchCmd, &stSrchRpy);
    PrintMatchTmplRpy(stSrchRpy);
    if (!stSrchRpy.matResultImg.empty())
        cv::imwrite("./data/TestSrchTmpl_7/SrchResult.png", stSrchRpy.matResultImg);
}

void TestTmplMatch_8() {
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl << "MATCH TEMPLATE REGRESSION TEST #8 STARTING";
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl;

    PR_LRN_TEMPLATE_CMD stLrnCmd;
    PR_LRN_TEMPLATE_RPY stLrnRpy;

    stLrnCmd.matInputImg = cv::imread("./data/TestSrchTmpl_8/Tmpl.png");
    stLrnCmd.enAlgorithm = PR_MATCH_TMPL_ALGORITHM::SQUARE_DIFF_NORMED;
    stLrnCmd.rectROI = cv::Rect(0, 0, stLrnCmd.matInputImg.cols, stLrnCmd.matInputImg.rows);

    PR_LrnTmpl(&stLrnCmd, &stLrnRpy);
    if (stLrnRpy.enStatus != VisionStatus::OK) {
        std::cout << "Failed to learn template." << std::endl;
        return;
    }

    PR_MATCH_TEMPLATE_CMD stSrchCmd;
    PR_MATCH_TEMPLATE_RPY stSrchRpy;
    stSrchCmd.matInputImg = cv::imread("./data/TestSrchTmpl_8/image.png");
    stSrchCmd.enAlgorithm = PR_MATCH_TMPL_ALGORITHM::SQUARE_DIFF_NORMED;
    stSrchCmd.enMotion = PR_OBJECT_MOTION::TRANSLATION;
    stSrchCmd.rectSrchWindow = cv::Rect(cv::Point(264, 166), cv::Point(792, 498));
    stSrchCmd.nRecordId = stLrnRpy.nRecordId;

    PR_MatchTmpl(&stSrchCmd, &stSrchRpy);
    PrintMatchTmplRpy(stSrchRpy);
    if (!stSrchRpy.matResultImg.empty())
        cv::imwrite("./data/TestSrchTmpl_8/SrchResult.png", stSrchRpy.matResultImg);
}

void TestTmplMatch() {
    TestTmplMatch_1();
    TestTmplMatch_2();
    TestTmplMatch_3();
    TestTmplMatch_4();
    TestTmplMatch_5();
    TestTmplMatch_6();
    TestTmplMatch_7();
    TestTmplMatch_8();
}

static void PrintFiducialMarkResult(const PR_SRCH_FIDUCIAL_MARK_RPY &stRpy) {
    std::cout << "Search fiducial status " << ToInt32(stRpy.enStatus) << std::endl;
    std::cout << "Match Score: " << ToInt32 ( stRpy.fMatchScore ) << std::endl;
    if ( VisionStatus::OK == stRpy.enStatus )
        std::cout << "Search fiducial result " << stRpy.ptPos.x << ", " << stRpy.ptPos.y << std::endl;
}

void TestSrchFiducialMark()
{
    std::cout << std::endl << "------------------------------------------------";
    std::cout << std::endl << "SEARCH FIDUCIAL MARK REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "------------------------------------------------";
    std::cout << std::endl;

    PR_SRCH_FIDUCIAL_MARK_CMD stCmd;
    PR_SRCH_FIDUCIAL_MARK_RPY stRpy;
    stCmd.enType = PR_FIDUCIAL_MARK_TYPE::SQUARE;
    stCmd.fSize = 54;
    stCmd.fMargin = 8;
    stCmd.matInputImg = cv::imread("./data/F6-313-1-Gray.bmp", cv::IMREAD_GRAYSCALE);
    stCmd.rectSrchWindow = cv::Rect(1459, 155, 500, 500);

    PR_SrchFiducialMark(&stCmd, &stRpy);
    PrintFiducialMarkResult(stRpy);

    std::cout << std::endl << "------------------------------------------------";
    std::cout << std::endl << "SEARCH FIDUCIAL MARK REGRESSION TEST #2 STARTING";
    std::cout << std::endl << "------------------------------------------------";
    std::cout << std::endl;

    stCmd.enType = PR_FIDUCIAL_MARK_TYPE::CIRCLE;
    stCmd.fSize = 64;
    stCmd.fMargin = 8;
    stCmd.matInputImg = cv::imread("./data/CircleFiducialMark.png", cv::IMREAD_GRAYSCALE);
    stCmd.rectSrchWindow = cv::Rect(0, 40, 250, 250);

    PR_SrchFiducialMark(&stCmd, &stRpy);
    PrintFiducialMarkResult(stRpy);

    std::cout << std::endl << "------------------------------------------------";
    std::cout << std::endl << "SEARCH FIDUCIAL MARK REGRESSION TEST #3 STARTING";
    std::cout << std::endl << "------------------------------------------------";
    std::cout << std::endl;
    stCmd.enType = PR_FIDUCIAL_MARK_TYPE::CIRCLE;
    stCmd.fSize = 64;
    stCmd.fMargin = 20;
    stCmd.matInputImg = cv::imread("./data/CircleFiducialMark.png", cv::IMREAD_GRAYSCALE);
    stCmd.rectSrchWindow = cv::Rect(126, 96, 200, 160);

    PR_SrchFiducialMark(&stCmd, &stRpy);
    PrintFiducialMarkResult(stRpy);
}

}
}