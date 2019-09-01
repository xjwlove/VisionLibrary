#include "stdafx.h"
#include "opencv2/highgui.hpp"
#include "../VisionLibrary/VisionAPI.h"
#include <iostream>
#include <iomanip>

using namespace AOI::Vision;

void TestMatchTmpl_1() {
    PR_LRN_TEMPLATE_CMD stLrnCmd;
    PR_LRN_TEMPLATE_RPY stLrnRpy;

    stLrnCmd.matInputImg = cv::imread("./data/TestOCV.png");
    stLrnCmd.enAlgorithm = PR_MATCH_TMPL_ALGORITHM::SQUARE_DIFF_NORMED;
    stLrnCmd.rectROI = cv::Rect(311, 818, 180, 61);
    cv::Mat matMask = cv::Mat::ones(stLrnCmd.rectROI.size(), CV_8UC1) * 255;
    cv::rectangle(matMask, cv::Rect(10, 10, 160, 30), cv::Scalar(0), CV_FILLED);
    //stLrnCmd.matMask = matMask;

    PR_LrnTmpl(&stLrnCmd, &stLrnRpy);

    PR_MATCH_TEMPLATE_CMD stSrchCmd;
    PR_MATCH_TEMPLATE_RPY stSrchRpy;
    stSrchCmd.matInputImg = cv::imread("./data/TestOCV.png");
    stSrchCmd.enAlgorithm = PR_MATCH_TMPL_ALGORITHM::SQUARE_DIFF_NORMED;
    stSrchCmd.enMotion = PR_OBJECT_MOTION::TRANSLATION;
    stSrchCmd.rectSrchWindow = cv::Rect(223, 1035, 339, 216);
    stSrchCmd.nRecordId = stLrnRpy.nRecordId;

    PR_MatchTmpl(&stSrchCmd, &stSrchRpy);

    std::cout << "PR_MatchTmpl status " << ToInt32(stSrchRpy.enStatus) << std::endl;
    if (stSrchRpy.enStatus != VisionStatus::OK)
        return;
    cv::imwrite("./data/MatchTmplResult.png", stSrchRpy.matResultImg);
}

void TestTmplMatch_9() {
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl << "MATCH TEMPLATE REGRESSION TEST #9 STARTING";
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl;

    PR_LRN_TEMPLATE_CMD stLrnCmd;
    PR_LRN_TEMPLATE_RPY stLrnRpy;

    stLrnCmd.matInputImg = cv::imread("./data/TestSrchTmpl_9/Tmpl.png");
    stLrnCmd.enAlgorithm = PR_MATCH_TMPL_ALGORITHM::SQUARE_DIFF_NORMED;
    stLrnCmd.rectROI = cv::Rect(0, 0, stLrnCmd.matInputImg.cols, stLrnCmd.matInputImg.rows);

    PR_LrnTmpl(&stLrnCmd, &stLrnRpy);
    if (stLrnRpy.enStatus != VisionStatus::OK) {
        std::cout << "Failed to learn template." << std::endl;
        return;
    }

    PR_MATCH_TEMPLATE_CMD stSrchCmd;
    PR_MATCH_TEMPLATE_RPY stSrchRpy;
    stSrchCmd.matInputImg = cv::imread("./data/TestSrchTmpl_9/image.png");
    stSrchCmd.enAlgorithm = PR_MATCH_TMPL_ALGORITHM::SQUARE_DIFF_NORMED;
    stSrchCmd.enMotion = PR_OBJECT_MOTION::TRANSLATION;
    stSrchCmd.rectSrchWindow = cv::Rect(0, 0, stSrchCmd.matInputImg.cols, stSrchCmd.matInputImg.rows);
    stSrchCmd.nRecordId = stLrnRpy.nRecordId;

    PR_MatchTmpl(&stSrchCmd, &stSrchRpy);
    //PrintMatchTmplRpy(stSrchRpy);
    if (!stSrchRpy.matResultImg.empty())
        cv::imwrite("./data/TestSrchTmpl_9/SrchResult.png", stSrchRpy.matResultImg);
}