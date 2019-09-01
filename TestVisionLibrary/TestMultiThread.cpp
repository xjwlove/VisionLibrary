#include "stdafx.h"
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <string>
#include <fstream>
#include <thread>
#include <atomic>
#include <iomanip>

#include "TestSub.h"

#include "../VisionLibrary/VisionAPI.h"
#include "opencv2/highgui.hpp"
#include "../RegressionTest/UtilityFunc.h"
#include "../VisionLibrary/CalcUtils.hpp"

using namespace AOI::Vision;

const std::string strParentFolder = "./data/TestInsp3DSolder/";
auto matHeight = readMatFromCsvFile(strParentFolder + "H3.csv");
auto matColorImg = cv::imread(strParentFolder + "image3.bmp", cv::IMREAD_COLOR);

static void TestInsp3DSolder() {
    const std::string strParentFolder = "./data/TestInsp3DSolder/";
    PR_INSP_3D_SOLDER_CMD stCmd;
    PR_INSP_3D_SOLDER_RPY stRpy;

    stCmd.matHeight = matHeight.clone();
    stCmd.matColorImg = matColorImg.clone();

    stCmd.rectDeviceROI = cv::Rect(1175, 575, 334, 179);
    stCmd.vecRectCheckROIs.push_back(cv::Rect(1197, 605, 72, 122));
    stCmd.vecRectCheckROIs.push_back(cv::Rect(1400, 605, 76, 122));

    //stCmd.rectDeviceROI = cv::Rect(1205, 1891, 302, 140);
    //stCmd.vecRectCheckROIs.push_back(cv::Rect(1218, 1897, 67, 108));
    //stCmd.vecRectCheckROIs.push_back(cv::Rect(1424, 1899, 63, 112));

    stCmd.nBaseColorDiff = 20;
    stCmd.nBaseGrayDiff = 20;
    
    cv::Rect rectBase(400, 200, 200, 200);
    cv::Mat matBaseRef(stCmd.matColorImg, rectBase);
    stCmd.scalarBaseColor = cv::mean(matBaseRef);

    PR_Insp3DSolder(&stCmd, &stRpy);

    std::cout << "PR_Insp3DSolder status " << ToInt32(stRpy.enStatus) << " at line " << __LINE__ << std::endl;
    if (VisionStatus::OK == stRpy.enStatus) {
        for (const auto &result : stRpy.vecResults) {
            std::cout << "Component height " << result.fComponentHeight << " Solder height " << result.fSolderHeight
                << ", area " << result.fSolderArea << ", ratio " << result.fSolderRatio << std::endl;
        }
    }
}

void TestMultipleThread() {
    std::atomic<int> nThread = 0;
    std::vector<std::unique_ptr<std::thread>> vecThread;
    for (int i = 0; i < 8; ++ i) {
        auto ptrThread = std::make_unique<std::thread>([&] {
            TestInsp3DSolder();
        });
        vecThread.push_back(std::move(ptrThread));
    }

    for (auto &ptrThread : vecThread)
        ptrThread->join();
}
