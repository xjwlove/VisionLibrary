#include "stdafx.h"

#include <iostream>
#include <iomanip>

#include "../VisionLibrary/VisionAPI.h"
#include "opencv2/highgui.hpp"
#include "TestSub.h"
#include "../RegressionTest/UtilityFunc.h"

using namespace AOI::Vision;

void WriteTableMappingResult(const std::string& filePath, const PR_TABLE_MAPPING_CMD& stCmd, const PR_TABLE_MAPPING_RPY& stRpy) {
    cv::FileStorage fs(filePath, cv::FileStorage::WRITE);
    if (!fs.isOpened())
        return;

    cv::write(fs, "MinX", stRpy.fMinX);
    cv::write(fs, "MaxX", stRpy.fMaxX);
    cv::write(fs, "MinY", stRpy.fMinY);
    cv::write(fs, "MaxY", stRpy.fMaxY);
    cv::write(fs, "a", stRpy.a);
    cv::write(fs, "b", stRpy.b);
    cv::write(fs, "c", stRpy.c);
    cv::write(fs, "XOffsetParam", stRpy.matXOffsetParam);
    cv::write(fs, "YOffsetParam", stRpy.matYOffsetParam);
    cv::write(fs, "BezierRankX", stCmd.nBezierRankX);
    cv::write(fs, "BezierRankY", stCmd.nBezierRankY);

    fs.release();
}

void TestTableMapping() {
    std::string strDataPath = "./data/TableMappingMergeTest/";
    const int TOTAL_FRAME = 1;
    const int DATA_START_ROW = 1;

    PR_TABLE_MAPPING_CMD stCmd;
    PR_TABLE_MAPPING_RPY stRpy;
    stCmd.nBezierRankX = 11;
    stCmd.nBezierRankY = 11;

    for (int i = 1; i <= TOTAL_FRAME; ++ i) {
        char chArrFileName[100];
        _snprintf(chArrFileName, sizeof(chArrFileName), "%02d.csv", i); 
        auto vecOfVecFloat = readDataFromFile(strDataPath + chArrFileName);

        PR_TABLE_MAPPING_CMD::VectorOfFramePoint vecFramePoint;
        vecFramePoint.reserve(vecOfVecFloat.size());

        for (int row = DATA_START_ROW; row < vecOfVecFloat.size(); ++ row) {
            PR_TABLE_MAPPING_CMD::FramePoint framePoint;
            framePoint.targetPoint = cv::Point2f(vecOfVecFloat[row][1], vecOfVecFloat[row][2]);
            framePoint.actualPoint = cv::Point2f(vecOfVecFloat[row][3], vecOfVecFloat[row][4]);
            vecFramePoint.push_back(framePoint);
        }

        stCmd.vecFramePoints.push_back(vecFramePoint);
    }

    PR_TableMapping(&stCmd, &stRpy);
    if (VisionStatus::OK != stRpy.enStatus)
        return;

    PR_CALC_TABLE_OFFSET_CMD stCalcOffsetCmd;
    PR_CALC_TABLE_OFFSET_RPY stCalcOffsetRpy;

    stCalcOffsetCmd.ptTablePos = cv::Point2f(50, 600);
    //stCalcOffsetCmd.ptTablePos = cv::Point2f(-300, 600);
    stCalcOffsetCmd.matXOffsetParam = stRpy.matXOffsetParam;
    stCalcOffsetCmd.matYOffsetParam = stRpy.matYOffsetParam;
    stCalcOffsetCmd.fMinX = stRpy.fMinX;
    stCalcOffsetCmd.fMaxX = stRpy.fMaxX;
    stCalcOffsetCmd.fMinY = stRpy.fMinY;
    stCalcOffsetCmd.fMaxY = stRpy.fMaxY;

    PR_CalcTableOffset(&stCalcOffsetCmd, &stCalcOffsetRpy);

    std::cout << "PR_CalcTableOffset status " << ToInt32(stCalcOffsetRpy.enStatus) << std::endl;
    std::cout << "Offset " << stCalcOffsetRpy.fOffsetX << ", " << stCalcOffsetRpy.fOffsetY <<std::endl;
}

void TestTableMapping_1() {
    std::string strDataPath = "./data/TableMappingTest_1/";
    const int TOTAL_FRAME = 1;
    const int DATA_START_ROW = 1;
    const int BEZIER_RANK = 5;

    PR_TABLE_MAPPING_CMD stCmd;
    PR_TABLE_MAPPING_RPY stRpy;
    stCmd.nBezierRankX = BEZIER_RANK;
    stCmd.nBezierRankY = BEZIER_RANK;
    stCmd.fBoardPointDist = 16;

    for (int i = 1; i <= TOTAL_FRAME; ++ i) {
        char chArrFileName[100];
        _snprintf(chArrFileName, sizeof(chArrFileName), "%02d.csv", i); 
        auto vecOfVecFloat = readDataFromFile(strDataPath + chArrFileName);

        PR_TABLE_MAPPING_CMD::VectorOfFramePoint vecFramePoint;
        vecFramePoint.reserve(vecOfVecFloat.size());

        for (int row = DATA_START_ROW; row < vecOfVecFloat.size(); ++ row) {
            PR_TABLE_MAPPING_CMD::FramePoint framePoint;
            framePoint.targetPoint = cv::Point2f(vecOfVecFloat[row][1], vecOfVecFloat[row][2]);
            framePoint.actualPoint = cv::Point2f(vecOfVecFloat[row][3], vecOfVecFloat[row][4]);
            vecFramePoint.push_back(framePoint);
        }

        stCmd.vecFramePoints.push_back(vecFramePoint);
    }

    PR_TableMapping(&stCmd, &stRpy);
    if (VisionStatus::OK != stRpy.enStatus)
        return;

    std::cout << "XOffsetParam" << std::endl;
    printfMat<float>(stRpy.matXOffsetParam, 3);
    std::cout << std::endl;

    std::cout << "YOffsetParam" << std::endl;
    printfMat<float>(stRpy.matYOffsetParam, 3);

    PR_CALC_TABLE_OFFSET_CMD stCalcOffsetCmd;
    PR_CALC_TABLE_OFFSET_RPY stCalcOffsetRpy;

    stCalcOffsetCmd.nBezierRankX = BEZIER_RANK;
    stCalcOffsetCmd.nBezierRankY = BEZIER_RANK;
    stCalcOffsetCmd.ptTablePos = cv::Point2f(-205, 103);
    //stCalcOffsetCmd.ptTablePos = cv::Point2f(-300, 600);
    stCalcOffsetCmd.matXOffsetParam = stRpy.matXOffsetParam;
    stCalcOffsetCmd.matYOffsetParam = stRpy.matYOffsetParam;
    stCalcOffsetCmd.fMinX = stRpy.fMinX;
    stCalcOffsetCmd.fMaxX = stRpy.fMaxX;
    stCalcOffsetCmd.fMinY = stRpy.fMinY;
    stCalcOffsetCmd.fMaxY = stRpy.fMaxY;

    PR_CalcTableOffset(&stCalcOffsetCmd, &stCalcOffsetRpy);

    std::cout << "PR_CalcTableOffset status " << ToInt32(stCalcOffsetRpy.enStatus) << std::endl;
    std::cout << "Offset " << stCalcOffsetRpy.fOffsetX << ", " << stCalcOffsetRpy.fOffsetY <<std::endl;
}

void TestTableMapping_2() {
    std::string strDataPath = "./data/TableMappingTest_2/";
    const int TOTAL_FRAME = 2;
    const int DATA_START_ROW = 1;
    const int BEZIER_RANK = 5;

    PR_TABLE_MAPPING_CMD stCmd;
    PR_TABLE_MAPPING_RPY stRpy;
    stCmd.nBezierRankX = BEZIER_RANK;
    stCmd.nBezierRankY = BEZIER_RANK;
    stCmd.fBoardPointDist = 16;
    stCmd.fFrameBorderPointWeight = 100.f;
    const int POINTS_PER_ROW = 13;

    for (int i = 1; i <= TOTAL_FRAME; ++ i) {
        char chArrFileName[100];
        _snprintf(chArrFileName, sizeof(chArrFileName), "%02d.csv", i); 
        auto vecOfVecFloat = readDataFromFile(strDataPath + chArrFileName);

        PR_TABLE_MAPPING_CMD::VectorOfFramePoint vecFramePoint;
        vecFramePoint.reserve(vecOfVecFloat.size());

        for (int row = DATA_START_ROW; row < vecOfVecFloat.size(); ++ row) {
            PR_TABLE_MAPPING_CMD::FramePoint framePoint;
            framePoint.targetPoint = cv::Point2f(vecOfVecFloat[row][1], vecOfVecFloat[row][2]);
            framePoint.actualPoint = cv::Point2f(vecOfVecFloat[row][3], vecOfVecFloat[row][4]);
            vecFramePoint.push_back(framePoint);
        }

        stCmd.vecFramePoints.push_back(vecFramePoint);
    }

    PR_TableMapping(&stCmd, &stRpy);
    if (VisionStatus::OK != stRpy.enStatus)
        return;

    std::cout << "XOffsetParam" << std::endl;
    printfMat<float>(stRpy.matXOffsetParam, 4);
    std::cout << std::endl;

    std::cout << "YOffsetParam" << std::endl;
    printfMat<float>(stRpy.matYOffsetParam, 4);

    PR_CALC_TABLE_OFFSET_CMD stCalcOffsetCmd;
    PR_CALC_TABLE_OFFSET_RPY stCalcOffsetRpy;

    stCalcOffsetCmd.nBezierRankX = BEZIER_RANK;
    stCalcOffsetCmd.nBezierRankY = BEZIER_RANK;
    stCalcOffsetCmd.ptTablePos = cv::Point2f(-205, 103);
    //stCalcOffsetCmd.ptTablePos = cv::Point2f(-300, 600);
    stCalcOffsetCmd.matXOffsetParam = stRpy.matXOffsetParam;
    stCalcOffsetCmd.matYOffsetParam = stRpy.matYOffsetParam;
    stCalcOffsetCmd.fMinX = stRpy.fMinX;
    stCalcOffsetCmd.fMaxX = stRpy.fMaxX;
    stCalcOffsetCmd.fMinY = stRpy.fMinY;
    stCalcOffsetCmd.fMaxY = stRpy.fMaxY;
    stCalcOffsetCmd.a = stRpy.a;
    stCalcOffsetCmd.b = stRpy.b;
    stCalcOffsetCmd.c = stRpy.c;

    PR_CalcTableOffset(&stCalcOffsetCmd, &stCalcOffsetRpy);

    std::cout << "PR_CalcTableOffset status " << ToInt32(stCalcOffsetRpy.enStatus) << std::endl;
    std::cout << "Offset " << stCalcOffsetRpy.fOffsetX << ", " << stCalcOffsetRpy.fOffsetY <<std::endl;
}

void TestTableMapping_3() {
    std::string strDataPath = "./data/TableMappingTest_3/";
    const int TOTAL_FRAME = 4;
    const int DATA_START_ROW = 1;
    const int BEZIER_RANK = 7;

    PR_TABLE_MAPPING_CMD stCmd;
    PR_TABLE_MAPPING_RPY stRpy;
    stCmd.nBezierRankX = BEZIER_RANK;
    stCmd.nBezierRankY = BEZIER_RANK;
    stCmd.fBoardPointDist = 16;
    stCmd.fFrameBorderPointWeight = 100.f;
    const int POINTS_PER_ROW = 13;

    for (int i = 1; i <= TOTAL_FRAME; ++i) {
        char chArrFileName[100];
        _snprintf(chArrFileName, sizeof(chArrFileName), "%d.csv", i);
        auto vecOfVecFloat = readDataFromFile(strDataPath + chArrFileName);

        PR_TABLE_MAPPING_CMD::VectorOfFramePoint vecFramePoint;
        vecFramePoint.reserve(vecOfVecFloat.size());

        for (int row = DATA_START_ROW; row < vecOfVecFloat.size(); ++row) {
            PR_TABLE_MAPPING_CMD::FramePoint framePoint;
            int index = row - DATA_START_ROW;
            framePoint.targetPoint = cv::Point2f(vecOfVecFloat[row][1], vecOfVecFloat[row][2]);
            framePoint.actualPoint = cv::Point2f(vecOfVecFloat[row][3], vecOfVecFloat[row][4]);
            vecFramePoint.push_back(framePoint);
        }

        stCmd.vecFramePoints.push_back(vecFramePoint);
    }

    PR_TableMapping(&stCmd, &stRpy);
    if (VisionStatus::OK != stRpy.enStatus)
        return;

    std::cout << "XOffsetParam" << std::endl;
    printfMat<float>(stRpy.matXOffsetParam, 4);
    std::cout << std::endl;

    std::cout << "YOffsetParam" << std::endl;
    printfMat<float>(stRpy.matYOffsetParam, 4);

    WriteTableMappingResult(strDataPath + "TableMapping.yml", stCmd, stRpy);

    PR_CALC_TABLE_OFFSET_CMD stCalcOffsetCmd;
    PR_CALC_TABLE_OFFSET_RPY stCalcOffsetRpy;

    stCalcOffsetCmd.nBezierRankX = BEZIER_RANK;
    stCalcOffsetCmd.nBezierRankY = BEZIER_RANK;
    stCalcOffsetCmd.ptTablePos = cv::Point2f(-205, 103);
    //stCalcOffsetCmd.ptTablePos = cv::Point2f(-300, 600);
    stCalcOffsetCmd.matXOffsetParam = stRpy.matXOffsetParam;
    stCalcOffsetCmd.matYOffsetParam = stRpy.matYOffsetParam;
    stCalcOffsetCmd.fMinX = stRpy.fMinX;
    stCalcOffsetCmd.fMaxX = stRpy.fMaxX;
    stCalcOffsetCmd.fMinY = stRpy.fMinY;
    stCalcOffsetCmd.fMaxY = stRpy.fMaxY;
    stCalcOffsetCmd.a = stRpy.a;
    stCalcOffsetCmd.b = stRpy.b;
    stCalcOffsetCmd.c = stRpy.c;

    PR_CalcTableOffset(&stCalcOffsetCmd, &stCalcOffsetRpy);

    std::cout << "PR_CalcTableOffset status " << ToInt32(stCalcOffsetRpy.enStatus) << std::endl;
    std::cout << "Offset " << stCalcOffsetRpy.fOffsetX << ", " << stCalcOffsetRpy.fOffsetY << std::endl;
}
