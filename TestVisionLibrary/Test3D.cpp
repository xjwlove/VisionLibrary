#include "stdafx.h"
#include "../VisionLibrary/VisionAPI.h"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <string>
#include "TestSub.h"

void TestCalc3DHeightDiff(const cv::Mat &matHeight);

using namespace AOI::Vision;

VectorOfFloat split ( const std::string &s, char delim ) {
    std::vector<float> elems;
    std::stringstream ss ( s );
    std::string strItem;
    while( std::getline ( ss, strItem, delim ) ) {
        elems.push_back ( ToFloat ( std::atof ( strItem.c_str() ) ) );
    }
    return elems;
}

VectorOfVectorOfFloat parseData(const std::string &strContent) {
    std::stringstream ss ( strContent );
    std::string strLine;
    VectorOfVectorOfFloat vecVecResult;
    while( std::getline ( ss, strLine ) ) {
        vecVecResult.push_back ( split ( strLine, ',' ) );
    }
    return vecVecResult;
}

VectorOfVectorOfFloat readDataFromFile(const std::string &strFilePath) {
    /* Open input file */
    FILE *fp;
    fopen_s ( &fp, strFilePath.c_str(), "r" );
    if(fp == NULL) {
        return VectorOfVectorOfFloat();
    }
    /* Get file size */
    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    rewind(fp);

    /* Allocate buffer and read */
    char *buff = new char[size + 1];
    size_t bytes = fread(buff, 1, size, fp);
    buff[bytes] = '\0';
    std::string strContent(buff);
    delete []buff;
    fclose ( fp );
    return parseData ( strContent );
}

static cv::Mat _drawHeightGrid(const cv::Mat &matHeight, int nGridRow, int nGridCol, const cv::Size &szMeasureWinSize = cv::Size(40, 40)) {
    double dMinValue = 0, dMaxValue = 0;
    cv::Mat matMask = ( matHeight == matHeight );
    cv::minMaxIdx ( matHeight, &dMinValue, &dMaxValue, 0, 0, matMask );
    
    cv::Mat matNewPhase = matHeight - dMinValue;

    float dRatio = 255.f / ToFloat( dMaxValue - dMinValue );
    matNewPhase = matNewPhase * dRatio;

    cv::Mat matResultImg;
    matNewPhase.convertTo ( matResultImg, CV_8UC1);
    cv::cvtColor ( matResultImg, matResultImg, CV_GRAY2BGR );

    int ROWS = matNewPhase.rows;
    int COLS = matNewPhase.cols;
    int nIntervalX = matNewPhase.cols / nGridCol;
    int nIntervalY = matNewPhase.rows / nGridRow;

    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 1;
    int thickness = 2;
    for ( int j = 0; j < nGridRow; ++ j )
    for ( int i = 0; i < nGridCol; ++ i ) {
        cv::Rect rectROI ( i * nIntervalX + nIntervalX / 2 - szMeasureWinSize.width  / 2,
                           j * nIntervalY + nIntervalY / 2 - szMeasureWinSize.height / 2,
                           szMeasureWinSize.width, szMeasureWinSize.height );
        cv::rectangle ( matResultImg, rectROI, cv::Scalar ( 0, 255, 0 ), 3 );

        cv::Mat matROI ( matHeight, rectROI );
        cv::Mat matMask = (matROI == matROI);
        float fAverage = ToFloat ( cv::mean ( matROI, matMask )[0] );

        char strAverage[100];
        _snprintf ( strAverage, sizeof(strAverage), "%.4f", fAverage );
        int baseline = 0;
        cv::Size textSize = cv::getTextSize ( strAverage, fontFace, fontScale, thickness, &baseline );
        //The height use '+' because text origin start from left-bottom.
        cv::Point ptTextOrg ( rectROI.x + (rectROI.width - textSize.width) / 2, rectROI.y + (rectROI.height + textSize.height) / 2 );
        cv::putText ( matResultImg, strAverage, ptTextOrg, fontFace, fontScale, cv::Scalar ( 255, 0, 0 ), thickness );
    }

    int nGridLineSize = 3;
    cv::Scalar scalarCyan(255, 255, 0);
    for ( int i = 1; i < nGridCol; ++ i )
        cv::line ( matResultImg, cv::Point(i * nIntervalX, 0), cv::Point(i * nIntervalX, ROWS), scalarCyan, nGridLineSize );
    for ( int i = 1; i < nGridRow; ++ i )
        cv::line ( matResultImg, cv::Point(0, i * nIntervalY), cv::Point(COLS, i * nIntervalY), scalarCyan, nGridLineSize );

    return matResultImg;
}

//static std::string gstrCalibResultFile("./data/capture/CalibPP.yml");
static std::string gstrWorkingFolder("./data/New3DCalibMethod_1/");
static std::string gstrCalibResultFile = gstrWorkingFolder + "CalibPP.yml";
void TestCalib3dBase() {
    const int IMAGE_COUNT = 12;
    std::string strFolder = gstrWorkingFolder + "0920234214_base/";
    PR_CALIB_3D_BASE_CMD stCmd;
    PR_CALIB_3D_BASE_RPY stRpy;
    for ( int i = 1; i <= IMAGE_COUNT; ++ i ) {
        char chArrFileName[100];
        _snprintf( chArrFileName, sizeof (chArrFileName), "%02d.bmp", i );
        std::string strImageFile = strFolder + chArrFileName;
        cv::Mat mat = cv::imread ( strImageFile, cv::IMREAD_GRAYSCALE );
        if ( mat.empty() ) {
            std::cout << "Failed to read image " << strImageFile << std::endl;
            return;
        }
        stCmd.vecInputImgs.push_back ( mat );
    }
    stCmd.bEnableGaussianFilter = true;
    stCmd.bReverseSeq = true;
    stCmd.fRemoveHarmonicWaveK = 0.f;
    PR_Calib3DBase ( &stCmd, &stRpy );
    std::cout << "PR_Calib3DBase status " << ToInt32( stRpy.enStatus ) << std::endl;

    std::string strResultMatPath = gstrCalibResultFile;
    cv::FileStorage fs(strResultMatPath, cv::FileStorage::WRITE);
    if ( ! fs.isOpened() )
        return;

    write ( fs, "K1", stRpy.matThickToThinK );
    write ( fs, "K2", stRpy.matThickToThinnestK );
    write ( fs, "BaseWrappedAlpha", stRpy.matBaseWrappedAlpha );
    write ( fs, "BaseWrappedBeta",  stRpy.matBaseWrappedBeta );
    write ( fs, "BaseWrappedGamma",  stRpy.matBaseWrappedGamma );
    fs.release();
}

void TestCalib3DHeight_01() {
    const int IMAGE_COUNT = 8;
    std::string strFolder = gstrWorkingFolder + "0920235040_lefttop/";
    PR_CALIB_3D_HEIGHT_CMD stCmd;
    PR_CALIB_3D_HEIGHT_RPY stRpy;
    for ( int i = 1; i <= IMAGE_COUNT; ++ i ) {
        char chArrFileName[100];
        _snprintf( chArrFileName, sizeof (chArrFileName), "%02d.bmp", i );
        std::string strImageFile = strFolder + chArrFileName;
        cv::Mat mat = cv::imread ( strImageFile, cv::IMREAD_GRAYSCALE );
        stCmd.vecInputImgs.push_back ( mat );
    }
    stCmd.bEnableGaussianFilter = true;
    stCmd.bReverseSeq = true;
    stCmd.bReverseHeight = false;
    stCmd.fMinAmplitude = 1.5;
    stCmd.nBlockStepCount = 5;
    stCmd.fBlockStepHeight = 1.f;
    stCmd.nResultImgGridRow = 10;
    stCmd.nResultImgGridCol = 10;
    cv::Mat matBaseSurfaceParam;

    std::string strResultMatPath = gstrCalibResultFile;
    cv::FileStorage fs ( strResultMatPath, cv::FileStorage::READ );
    cv::FileNode fileNode = fs["K1"];
    cv::read ( fileNode, stCmd.matThickToThinK, cv::Mat() );
    fileNode = fs["K2"];
    cv::read ( fileNode, stCmd.matThickToThinnestK, cv::Mat() );
    fileNode = fs["BaseWrappedAlpha"];
    cv::read ( fileNode, stCmd.matBaseWrappedAlpha, cv::Mat() );
    fileNode = fs["BaseWrappedBeta"];
    cv::read ( fileNode, stCmd.matBaseWrappedBeta, cv::Mat() );
    fileNode = fs["BaseWrappedGamma"];
    cv::read ( fileNode, stCmd.matBaseWrappedGamma, cv::Mat() );
    fs.release();

    PR_Calib3DHeight ( &stCmd, &stRpy );
    std::cout << "PR_Calib3DHeight status " << ToInt32( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK != stRpy.enStatus )
        return;

    cv::imwrite ( gstrWorkingFolder + "DivideStepResultImg_01.png", stRpy.matDivideStepResultImg );
    cv::imwrite ( gstrWorkingFolder + "Calib3DHeightResultImg_01.png", stRpy.matResultImg );

    cv::FileStorage fs1(strResultMatPath, cv::FileStorage::APPEND);
    if ( ! fs1.isOpened() )
        return;
    cv::write ( fs1, "PhaseToHeightK", stRpy.matPhaseToHeightK );
    fs1.release();

    //Write the phase and divide step result for PR_Integrate3DCalib.
    std::string strCalibDataFile(gstrWorkingFolder + "01.yml");
    cv::FileStorage fsCalibData ( strCalibDataFile, cv::FileStorage::WRITE );
    if ( ! fsCalibData.isOpened() )
        return;
    cv::write ( fsCalibData, "Phase", stRpy.matPhase );
    cv::write ( fsCalibData, "DivideStepIndex", stRpy.matDivideStepIndex );
    fsCalibData.release();
}

void TestCalib3DHeight_02() {
    const int IMAGE_COUNT = 8;
    std::string strFolder = gstrWorkingFolder + "0920235128_rightbottom/";
    PR_CALIB_3D_HEIGHT_CMD stCmd;
    PR_CALIB_3D_HEIGHT_RPY stRpy;
    for ( int i = 1; i <= IMAGE_COUNT; ++ i ) {
        char chArrFileName[100];
        _snprintf( chArrFileName, sizeof (chArrFileName), "%02d.bmp", i );
        std::string strImageFile = strFolder + chArrFileName;
        cv::Mat mat = cv::imread ( strImageFile, cv::IMREAD_GRAYSCALE );
        stCmd.vecInputImgs.push_back ( mat );
    }
    stCmd.bEnableGaussianFilter = true;
    stCmd.bReverseSeq = true;
    stCmd.bReverseHeight = false;
    stCmd.fMinAmplitude = 1.5;
    stCmd.nBlockStepCount = 5;
    stCmd.fBlockStepHeight = 1.f;
    stCmd.nResultImgGridRow = 10;
    stCmd.nResultImgGridCol = 10;

    std::string strResultMatPath = gstrCalibResultFile;
    cv::FileStorage fs ( strResultMatPath, cv::FileStorage::READ );
    cv::FileNode fileNode = fs["K1"];
    cv::read ( fileNode, stCmd.matThickToThinK, cv::Mat() );
    fileNode = fs["K2"];
    cv::read ( fileNode, stCmd.matThickToThinnestK, cv::Mat() );
    fileNode = fs["BaseWrappedAlpha"];
    cv::read ( fileNode, stCmd.matBaseWrappedAlpha, cv::Mat() );
    fileNode = fs["BaseWrappedBeta"];
    cv::read ( fileNode, stCmd.matBaseWrappedBeta, cv::Mat() );
    fileNode = fs["BaseWrappedGamma"];
    cv::read ( fileNode, stCmd.matBaseWrappedGamma, cv::Mat() );
    fs.release();

    PR_Calib3DHeight ( &stCmd, &stRpy );
    std::cout << "PR_Calib3DHeight status " << ToInt32( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK != stRpy.enStatus )
        return;

    cv::imwrite ( gstrWorkingFolder + "DivideStepResultImg_02.png", stRpy.matDivideStepResultImg );
    cv::imwrite ( gstrWorkingFolder + "Calib3DHeightResultImg_02.png", stRpy.matResultImg );

    cv::FileStorage fs1(strResultMatPath, cv::FileStorage::APPEND);
    if ( ! fs1.isOpened() )
        return;
    cv::write ( fs1, "PhaseToHeightK", stRpy.matPhaseToHeightK );
    fs1.release();

    //Write the phase and divide step result for PR_Integrate3DCalib.
    std::string strCalibDataFile(gstrWorkingFolder + "02.yml");
    cv::FileStorage fsCalibData ( strCalibDataFile, cv::FileStorage::WRITE );
    if ( ! fsCalibData.isOpened() )
        return;
    cv::write ( fsCalibData, "Phase", stRpy.matPhase );
    cv::write ( fsCalibData, "DivideStepIndex", stRpy.matDivideStepIndex );
    fsCalibData.release();
}

void TestCalib3DHeight_03() {
    const int IMAGE_COUNT = 8;
    std::string strFolder = gstrWorkingFolder + "0920235405_negitive/";
    PR_CALIB_3D_HEIGHT_CMD stCmd;
    PR_CALIB_3D_HEIGHT_RPY stRpy;
    for ( int i = 1; i <= IMAGE_COUNT; ++ i ) {
        char chArrFileName[100];
        _snprintf( chArrFileName, sizeof (chArrFileName), "%02d.bmp", i );
        std::string strImageFile = strFolder + chArrFileName;
        cv::Mat mat = cv::imread ( strImageFile, cv::IMREAD_GRAYSCALE );
        stCmd.vecInputImgs.push_back ( mat );
    }
    stCmd.bEnableGaussianFilter = true;
    stCmd.bReverseSeq = true;
    stCmd.bReverseHeight = true;
    stCmd.fMinAmplitude = 1.5;
    stCmd.nBlockStepCount = 3;
    stCmd.fBlockStepHeight = 1.f;
    stCmd.nResultImgGridRow = 10;
    stCmd.nResultImgGridCol = 10;
    cv::Mat matBaseSurfaceParam;

    std::string strResultMatPath = gstrCalibResultFile;
    cv::FileStorage fs ( strResultMatPath, cv::FileStorage::READ );
    cv::FileNode fileNode = fs["K1"];
    cv::read ( fileNode, stCmd.matThickToThinK, cv::Mat() );
    fileNode = fs["K2"];
    cv::read ( fileNode, stCmd.matThickToThinnestK, cv::Mat() );
    fileNode = fs["BaseWrappedAlpha"];
    cv::read ( fileNode, stCmd.matBaseWrappedAlpha, cv::Mat() );
    fileNode = fs["BaseWrappedBeta"];
    cv::read ( fileNode, stCmd.matBaseWrappedBeta, cv::Mat() );
    fileNode = fs["BaseWrappedGamma"];
    cv::read ( fileNode, stCmd.matBaseWrappedGamma, cv::Mat() );
    fs.release();

    PR_Calib3DHeight ( &stCmd, &stRpy );
    std::cout << "PR_Calib3DHeight status " << ToInt32( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK != stRpy.enStatus )
        return;

    cv::imwrite ( gstrWorkingFolder + "DivideStepResultImg_03.png", stRpy.matDivideStepResultImg );
    cv::imwrite ( gstrWorkingFolder + "Calib3DHeightResultImg_03.png", stRpy.matResultImg );

    cv::FileStorage fs1(strResultMatPath, cv::FileStorage::APPEND);
    if ( ! fs1.isOpened() )
        return;
    cv::write ( fs1, "PhaseToHeightK", stRpy.matPhaseToHeightK );
    fs1.release();

    //Write the phase and divide step result for PR_Integrate3DCalib.
    std::string strCalibDataFile(gstrWorkingFolder + "03.yml");
    cv::FileStorage fsCalibData ( strCalibDataFile, cv::FileStorage::WRITE );
    if ( ! fsCalibData.isOpened() )
        return;
    cv::write ( fsCalibData, "Phase", stRpy.matPhase );
    cv::write ( fsCalibData, "DivideStepIndex", stRpy.matDivideStepIndex );
    fsCalibData.release();
}

void TestCalc3DHeight() {
    const int IMAGE_COUNT = 8;
    std::string strFolder = gstrWorkingFolder + "0920235040_lefttop/";
    //std::string strFolder = "./data/0913212217_Unwrap_Not_Finish/";
    PR_CALC_3D_HEIGHT_CMD stCmd;
    PR_CALC_3D_HEIGHT_RPY stRpy;
    for ( int i = 1; i <= IMAGE_COUNT; ++ i ) {
        char chArrFileName[100];
        _snprintf( chArrFileName, sizeof (chArrFileName), "%02d.bmp", i );
        std::string strImageFile = strFolder + chArrFileName;
        cv::Mat mat = cv::imread ( strImageFile, cv::IMREAD_GRAYSCALE );
        stCmd.vecInputImgs.push_back ( mat );
    }
    stCmd.bEnableGaussianFilter = true;
    stCmd.bReverseSeq = true;
    stCmd.fMinAmplitude = 1.5;

    cv::Mat matBaseSurfaceParam;

    std::string strResultMatPath = gstrCalibResultFile;
    cv::FileStorage fs ( strResultMatPath, cv::FileStorage::READ );
    cv::FileNode fileNode = fs["K1"];
    cv::read ( fileNode, stCmd.matThickToThinK, cv::Mat() );
    fileNode = fs["K2"];
    cv::read ( fileNode, stCmd.matThickToThinnestK, cv::Mat() );
    fileNode = fs["BaseWrappedAlpha"];
    cv::read ( fileNode, stCmd.matBaseWrappedAlpha, cv::Mat() );
    fileNode = fs["BaseWrappedBeta"];
    cv::read ( fileNode, stCmd.matBaseWrappedBeta, cv::Mat() );
    fileNode = fs["BaseWrappedGamma"];
    cv::read ( fileNode, stCmd.matBaseWrappedGamma, cv::Mat() );
    fs.release();

    PR_Calc3DHeight ( &stCmd, &stRpy );
    std::cout << "PR_Calc3DHeight status " << ToInt32( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK != stRpy.enStatus )
        return;

    cv::Mat matHeightResultImg = _drawHeightGrid ( stRpy.matHeight, 9, 9 );
    cv::imwrite ( gstrWorkingFolder + "PR_Calc3DHeight_HeightGridImg.png", matHeightResultImg );
    cv::Mat matPhaseResultImg = _drawHeightGrid ( stRpy.matPhase, 9, 9 );
    cv::imwrite ( gstrWorkingFolder + "PR_Calc3DHeight_PhaseGridImg.png", matPhaseResultImg );

    //std::string strCalibDataFile(gstrWorkingFolder + "H5.yml");
    //cv::FileStorage fsCalibData ( strCalibDataFile, cv::FileStorage::WRITE );
    //if ( ! fsCalibData.isOpened() )
    //    return;
    //cv::write ( fsCalibData, "Phase", stRpy.matPhase );
    //fsCalibData.release();
}

void TestComb3DCalib() {
    PR_COMB_3D_CALIB_CMD stCmd;
    PR_COMB_3D_CALIB_RPY stRpy;
    stCmd.vecVecStepPhaseNeg = readDataFromFile("./data/StepPhaseData/StepPhaseNeg.csv");
    stCmd.vecVecStepPhasePos = readDataFromFile("./data/StepPhaseData/StepPhasePos.csv");
    PR_Comb3DCalib ( &stCmd, &stRpy );
    std::cout << "PR_Comb3DCalib status " << ToInt32( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK == stRpy.enStatus ) {
        std::cout << "Result PhaseToHeightK " << std::endl;
        printfMat<float>(stRpy.matPhaseToHeightK);
        std::cout << "Step Phase Difference: " << std::endl;
        printfVectorOfVector<float>(stRpy.vecVecStepPhaseDiff );
    }
}

void TestIntegrate3DCalib() {
    PR_INTEGRATE_3D_CALIB_CMD stCmd;
    PR_INTEGRATE_3D_CALIB_RPY stRpy;
    for ( int i = 1; i <= 3; ++ i ) {
        char chArrFileName[100];
        _snprintf( chArrFileName, sizeof (chArrFileName), "%02d.yml", i );
        std::string strDataFile = gstrWorkingFolder + chArrFileName;
        
        std::string strCalibDataFile( strDataFile );
        cv::FileStorage fsCalibData ( strCalibDataFile, cv::FileStorage::READ );
        if ( ! fsCalibData.isOpened() ) {
            std::cout << "Failed to open file: " << strCalibDataFile << std::endl;
            return;
        }

        PR_INTEGRATE_3D_CALIB_CMD::SINGLE_CALIB_DATA stCalibData;
        cv::FileNode fileNode = fsCalibData["Phase"];
        cv::read ( fileNode, stCalibData.matPhase, cv::Mat() );

        fileNode = fsCalibData["DivideStepIndex"];
        cv::read ( fileNode, stCalibData.matDivideStepIndex, cv::Mat() );
        fsCalibData.release();

        stCmd.vecCalibData.push_back ( stCalibData );
    }

    std::string strCalibDataFile( gstrWorkingFolder + "H5.yml" );
    cv::FileStorage fsCalibData ( strCalibDataFile, cv::FileStorage::READ );
    if (!fsCalibData.isOpened ()) {
        std::cout << "Failed to open file: " << strCalibDataFile << std::endl;
        return;
    }
    cv::FileNode fileNode = fsCalibData["Phase"];
    cv::read ( fileNode, stCmd.matTopSurfacePhase, cv::Mat() );
    fsCalibData.release();

    stCmd.fTopSurfaceHeight = 5;

    PR_Integrate3DCalib ( &stCmd, &stRpy );
    std::cout << "PR_Integrate3DCalib status " << ToInt32( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK != stRpy.enStatus )
        return;

    int i = 1;
    for ( const auto &matResultImg : stRpy.vecMatResultImg ) {
        char chArrFileName[100];
        _snprintf( chArrFileName, sizeof (chArrFileName), "ResultImg_%02d.png", i );
        std::string strDataFile = gstrWorkingFolder + chArrFileName;
        cv::imwrite ( strDataFile, matResultImg );
        ++ i;
    }

    std::string strCalibResultFile( gstrWorkingFolder + "IntegrateCalibResult.yml" );
    cv::FileStorage fsCalibResultData ( strCalibResultFile, cv::FileStorage::WRITE );
    if (!fsCalibResultData.isOpened ()) {
        std::cout << "Failed to open file: " << strCalibResultFile << std::endl;
        return;
    }
    cv::write ( fsCalibResultData, "IntegratedK", stRpy.matIntegratedK );
    cv::write ( fsCalibResultData, "Order3CurveSurface", stRpy.matOrder3CurveSurface );
    fsCalibResultData.release();
}

void TestIntegrate3DCalibHaoYu() {
    PR_INTEGRATE_3D_CALIB_CMD stCmd;
    PR_INTEGRATE_3D_CALIB_RPY stRpy;
    for ( int i = 1; i <= 3; ++ i ) {
        char chArrFileName[100];
        _snprintf( chArrFileName, sizeof (chArrFileName), "%d.yml", i );
        std::string strDataFile = gstrWorkingFolder + chArrFileName;
        
        std::string strCalibDataFile( strDataFile );
        cv::FileStorage fsCalibData ( strCalibDataFile, cv::FileStorage::READ );
        if ( ! fsCalibData.isOpened() ) {
            std::cout << "Failed to open file: " << strCalibDataFile << std::endl;
            return;
        }

        PR_INTEGRATE_3D_CALIB_CMD::SINGLE_CALIB_DATA stCalibData;
        cv::FileNode fileNode = fsCalibData["Phase"];
        cv::read ( fileNode, stCalibData.matPhase, cv::Mat() );

        fileNode = fsCalibData["DivideStepIndex"];
        cv::read ( fileNode, stCalibData.matDivideStepIndex, cv::Mat() );
        fsCalibData.release();

        stCmd.vecCalibData.push_back ( stCalibData );
    }

    std::string strCalibDataFile( gstrWorkingFolder + "H5.yml" );
    cv::FileStorage fsCalibData ( strCalibDataFile, cv::FileStorage::READ );
    if (!fsCalibData.isOpened ()) {
        std::cout << "Failed to open file: " << strCalibDataFile << std::endl;
        return;
    }
    cv::FileNode fileNode = fsCalibData["Phase"];
    cv::read ( fileNode, stCmd.matTopSurfacePhase, cv::Mat() );
    fsCalibData.release();

    stCmd.fTopSurfaceHeight = 5;

    PR_Integrate3DCalib ( &stCmd, &stRpy );
    std::cout << "PR_Integrate3DCalib status " << ToInt32( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK != stRpy.enStatus )
        return;

    int i = 1;
    for ( const auto &matResultImg : stRpy.vecMatResultImg ) {
        char chArrFileName[100];
        _snprintf( chArrFileName, sizeof (chArrFileName), "ResultImg_%02d.png", i );
        std::string strDataFile = gstrWorkingFolder + chArrFileName;
        cv::imwrite ( strDataFile, matResultImg );
        ++ i;
    }

    std::string strCalibResultFile( gstrWorkingFolder + "IntegrateCalibResult.yml" );
    cv::FileStorage fsCalibResultData ( strCalibResultFile, cv::FileStorage::WRITE );
    if (!fsCalibResultData.isOpened ()) {
        std::cout << "Failed to open file: " << strCalibResultFile << std::endl;
        return;
    }
    cv::write ( fsCalibResultData, "IntegratedK", stRpy.matIntegratedK );
    cv::write ( fsCalibResultData, "Order3CurveSurface", stRpy.matOrder3CurveSurface );
    fsCalibResultData.release();
}

void TestCalc3DHeightNew() {
    //const int IMAGE_COUNT = 8;
    //std::string strFolder = gstrWorkingFolder + "0920235128_rightbottom/";
    ////std::string strFolder = "./data/0913212217_Unwrap_Not_Finish/";
    //PR_CALC_3D_HEIGHT_CMD stCmd;
    //PR_CALC_3D_HEIGHT_RPY stRpy;
    //for ( int i = 1; i <= IMAGE_COUNT; ++ i ) {
    //    char chArrFileName[100];
    //    _snprintf( chArrFileName, sizeof (chArrFileName), "%02d.bmp", i );
    //    std::string strImageFile = strFolder + chArrFileName;
    //    cv::Mat mat = cv::imread ( strImageFile, cv::IMREAD_GRAYSCALE );
    //    stCmd.vecInputImgs.push_back ( mat );
    //}
    //stCmd.bEnableGaussianFilter = true;
    //stCmd.bReverseSeq = true;
    //stCmd.fMinIntensityDiff = 3;
    //stCmd.fMinAvgIntensity = 2;

    //cv::Mat matBaseSurfaceParam;
    //{
    //    std::string strResultMatPath = gstrCalibResultFile;
    //    cv::FileStorage fs ( strResultMatPath, cv::FileStorage::READ );
    //    cv::FileNode fileNode = fs["K"];
    //    cv::read ( fileNode, stCmd.matThickToThinStripeK, cv::Mat () );

    //    fileNode = fs["PPz"];
    //    cv::read ( fileNode, matBaseSurfaceParam, cv::Mat () );

    //    fileNode = fs["BaseStartAvgPhase"];
    //    cv::read ( fileNode, stCmd.fBaseStartAvgPhase, 0.f );

    //    fs.release ();
    //}    

    //PR_CALC_3D_BASE_CMD stCalc3DBaseCmd;
    //PR_CALC_3D_BASE_RPY stCalc3DBaseRpy;
    //stCalc3DBaseCmd.matBaseSurfaceParam = matBaseSurfaceParam;
    //PR_Calc3DBase ( &stCalc3DBaseCmd, &stCalc3DBaseRpy );
    //if ( VisionStatus::OK != stCalc3DBaseRpy.enStatus ) {
    //    std::cout << "PR_Calc3DBase fail. Status = " << ToInt32 ( stCalc3DBaseRpy.enStatus ) << std::endl;
    //    return;
    //}    
    //stCmd.matBaseSurface = stCalc3DBaseRpy.matBaseSurface;

    //{
    //    std::string strIntegratedCalibResultPath = gstrWorkingFolder + "IntegrateCalibResult.yml";
    //    cv::FileStorage fs ( strIntegratedCalibResultPath, cv::FileStorage::READ );
    //    cv::FileNode fileNode = fs["IntegratedK"];
    //    cv::read ( fileNode, stCmd.matIntegratedK, cv::Mat () );

    //    fileNode = fs["Order3CurveSurface"];
    //    cv::read ( fileNode, stCmd.matOrder3CurveSurface, cv::Mat () );
    //    fs.release ();
    //}

    //PR_Calc3DHeight ( &stCmd, &stRpy );
    //std::cout << "PR_Calc3DHeight status " << ToInt32( stRpy.enStatus ) << std::endl;
    //if ( VisionStatus::OK != stRpy.enStatus )
    //    return;

    //cv::Mat matHeightResultImg = _drawHeightGrid ( stRpy.matHeight, 10, 10 );
    //cv::imwrite ( gstrWorkingFolder + "PR_Calc3DHeight_HeightGridImg.png", matHeightResultImg );
    //cv::Mat matPhaseResultImg = _drawHeightGrid ( stRpy.matPhase, 10, 10 );
    //cv::imwrite ( gstrWorkingFolder + "PR_Calc3DHeight_PhaseGridImg.png", matPhaseResultImg );
}

void TestFastCalc3DHeight() {
    const int IMAGE_COUNT = 8;
    std::string strFolder = gstrWorkingFolder + "0920235128_rightbottom/";
    //std::string strFolder = "./data/0913212217_Unwrap_Not_Finish/";
    PR_FAST_CALC_3D_HEIGHT_CMD stCmd;
    PR_FAST_CALC_3D_HEIGHT_RPY stRpy;
    for ( int i = 1; i <= IMAGE_COUNT; ++ i ) {
        char chArrFileName[100];
        _snprintf( chArrFileName, sizeof (chArrFileName), "%02d.bmp", i );
        std::string strImageFile = strFolder + chArrFileName;
        cv::Mat mat = cv::imread ( strImageFile, cv::IMREAD_GRAYSCALE );
        stCmd.vecInputImgs.push_back ( mat );
    }
    stCmd.bEnableGaussianFilter = false;
    stCmd.bReverseSeq = true;
    stCmd.fMinAmplitude = 2;

    cv::Mat matBaseSurfaceParam;
    {
        std::string strResultMatPath = gstrCalibResultFile;
        cv::FileStorage fs ( strResultMatPath, cv::FileStorage::READ );
        cv::FileNode fileNode = fs["K"];
        cv::read ( fileNode, stCmd.matThickToThinStripeK, cv::Mat () );

        fileNode = fs["PPz"];
        cv::read ( fileNode, matBaseSurfaceParam, cv::Mat () );

        fileNode = fs["BaseWrappedAlpha"];
        cv::read ( fileNode, stCmd.matBaseWrappedAlpha, cv::Mat () );

        fileNode = fs["BaseWrappedBeta"];
        cv::read ( fileNode, stCmd.matBaseWrappedBeta, cv::Mat () );

        fs.release ();
    }

    {
        std::string strIntegratedCalibResultPath = gstrWorkingFolder + "IntegrateCalibResult.yml";
        cv::FileStorage fs ( strIntegratedCalibResultPath, cv::FileStorage::READ );
        cv::FileNode fileNode = fs["IntegratedK"];
        cv::read ( fileNode, stCmd.matIntegratedK, cv::Mat () );

        fileNode = fs["Order3CurveSurface"];
        cv::read ( fileNode, stCmd.matOrder3CurveSurface, cv::Mat () );
        fs.release ();
    }

    PR_FastCalc3DHeight ( &stCmd, &stRpy );
    std::cout << "PR_FastCalc3DHeight status " << ToInt32( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK != stRpy.enStatus )
        return;

    std::string strHeightResultPath = gstrWorkingFolder + "Height.yml";
    cv::FileStorage fs ( strHeightResultPath, cv::FileStorage::WRITE );
    cv::write ( fs, "Height", stRpy.matHeight );
    fs.release();

    cv::Mat matHeightResultImg = _drawHeightGrid ( stRpy.matHeight, 10, 10 );
    cv::imwrite ( gstrWorkingFolder + "PR_FastCalc3DHeight_HeightGridImg.png", matHeightResultImg );
    cv::Mat matPhaseResultImg = _drawHeightGrid ( stRpy.matPhase, 10, 10 );
    cv::imwrite ( gstrWorkingFolder + "PR_FastCalc3DHeight_PhaseGridImg.png", matPhaseResultImg );
}

void copyVectorOfVectorToMat(const VectorOfVectorOfFloat &vecVecInput, cv::Mat &matOutput) {
    matOutput = cv::Mat ( vecVecInput.size(), vecVecInput[0].size(), CV_32FC1 );
    for ( int row = 0; row < matOutput.rows; ++ row )
    for ( int col = 0; col < matOutput.cols; ++ col ) {
        matOutput.at<float>(row, col) = vecVecInput[row][col];
    }
}

static cv::Mat calcOrder3Surface(const cv::Mat &matX, const cv::Mat &matY, const cv::Mat &matK) {
    cv::Mat matXPow2 = matX.    mul ( matX );
    cv::Mat matXPow3 = matXPow2.mul ( matX );
    cv::Mat matYPow2 = matY.    mul ( matY );
    cv::Mat matYPow3 = matYPow2.mul ( matY );
    cv::Mat matArray[] = { 
        matXPow3 * matK.at<float>(0),
        matYPow3 * matK.at<float>(1),
        matXPow2.mul ( matY ) * matK.at<float>(2),
        matYPow2.mul ( matX ) * matK.at<float>(3),
        matXPow2 * matK.at<float>(4),
        matYPow2 * matK.at<float>(5),
        matX.mul ( matY ) * matK.at<float>(6),
        matX * matK.at<float>(7),
        matY * matK.at<float>(8),
        cv::Mat::ones ( matX.size(), matX.type() ) * matK.at<float>(9),
    };
    const int RANK_3_SURFACE_COEFF_COL = 10;
    static_assert ( sizeof( matArray ) / sizeof(cv::Mat) == RANK_3_SURFACE_COEFF_COL, "Size of array not correct" );
    cv::Mat matResult = matArray[0];
    for ( int i = 1; i < RANK_3_SURFACE_COEFF_COL; ++ i )
        matResult = matResult + matArray[i];
    return matResult;
}

template<typename _tp>
static cv::Mat intervals ( _tp start, _tp interval, _tp end ) {
    std::vector<_tp> vecValue;
    int nSize = ToInt32 ( (end - start) / interval );
    if (nSize <= 0) {
        static std::string msg = std::string ( __FUNCTION__ ) + " input paramters are invalid.";
        throw std::exception ( msg.c_str () );
    }
    vecValue.reserve ( nSize );
    _tp value = start;
    if (interval > 0) {
        while (value <= end) {
            vecValue.push_back ( value );
            value += interval;
        }
    }
    else {
        while (value >= end) {
            vecValue.push_back ( value );
            value += interval;
        }
    }
    //cv::Mat matResult ( vecValue ); //This have problem, because matResult share the memory with vecValue, after leave this function, the memory already released.
    return cv::Mat ( vecValue ).clone ();
}

template<typename _tp>
static void meshgrid ( _tp xStart, _tp xInterval, _tp xEnd, _tp yStart, _tp yInterval, _tp yEnd, cv::Mat &matX, cv::Mat &matY ) {
    cv::Mat matCol = intervals<_tp> ( xStart, xInterval, xEnd );
    matCol = matCol.reshape ( 1, 1 );

    cv::Mat matRow = intervals<_tp> ( yStart, yInterval, yEnd );
    matX = cv::repeat ( matCol, matRow.rows, 1 );
    matY = cv::repeat ( matRow, 1, matCol.cols );
}

void TestFastCalc3DHeight_1() {
    const int IMAGE_COUNT = 8;
    std::string strFolder = gstrWorkingFolder + "1027160744_01/";
    //std::string strFolder = "./data/0913212217_Unwrap_Not_Finish/";
    PR_FAST_CALC_3D_HEIGHT_CMD stCmd;
    PR_FAST_CALC_3D_HEIGHT_RPY stRpy;
    for ( int i = 1; i <= IMAGE_COUNT; ++ i ) {
        char chArrFileName[100];
        _snprintf( chArrFileName, sizeof (chArrFileName), "%02d.bmp", i );
        std::string strImageFile = strFolder + chArrFileName;
        cv::Mat mat = cv::imread ( strImageFile, cv::IMREAD_GRAYSCALE );
        stCmd.vecInputImgs.push_back ( mat );
    }
    stCmd.bEnableGaussianFilter = false;
    stCmd.bReverseSeq = true;
    stCmd.fMinAmplitude = 2.236;

    auto vecVecData = readDataFromFile ( gstrWorkingFolder + "CalibrationMatrix3/pAlpha.csv");
    copyVectorOfVectorToMat ( vecVecData, stCmd.matBaseWrappedAlpha );

    vecVecData = readDataFromFile ( gstrWorkingFolder + "CalibrationMatrix3/pBeta.csv");
    copyVectorOfVectorToMat ( vecVecData, stCmd.matBaseWrappedBeta );

    stCmd.matThickToThinStripeK = cv::Mat(2, 1, CV_32FC1);
    stCmd.matThickToThinStripeK.at<float>(0) = 4.999762f;    

    VectorOfFloat vecK{ -4.96231398e-012f, 1.05858230e-011f, 8.15840884e-012f,
        -4.87927320e-012f, 3.07154580e-010f, -2.42397249e-008f,
        1.17145926e-009f, -1.87926678e-004f, 2.95515900e-004f,
        2.17526913e+000f, -3.28916986e-003f, 1.85934448e-004f };
    stCmd.matIntegratedK = cv::Mat ( vecK );

    cv::Mat matX, matY;
    meshgrid<float> ( 1.f, 1.f, ToFloat ( stCmd.matBaseWrappedAlpha.cols ), 1.f, 1.f, ToFloat ( stCmd.matBaseWrappedAlpha.rows ), matX, matY );
    stCmd.matOrder3CurveSurface = calcOrder3Surface ( matX, matY, stCmd.matIntegratedK );

    PR_FastCalc3DHeight ( &stCmd, &stRpy );
    std::cout << "PR_FastCalc3DHeight status " << ToInt32( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK != stRpy.enStatus )
        return;

    std::string strHeightResultPath = gstrWorkingFolder + "Height_1.yml";
    cv::FileStorage fs ( strHeightResultPath, cv::FileStorage::WRITE );
    cv::write ( fs, "Height", stRpy.matHeight );
    fs.release();

    cv::Mat matHeightResultImg = _drawHeightGrid ( stRpy.matHeight, 10, 10 );
    cv::imwrite ( gstrWorkingFolder + "PR_FastCalc3DHeight_HeightGridImg.png", matHeightResultImg );
    cv::Mat matPhaseResultImg = _drawHeightGrid ( stRpy.matPhase, 10, 10 );
    cv::imwrite ( gstrWorkingFolder + "PR_FastCalc3DHeight_PhaseGridImg.png", matPhaseResultImg );
}

void TestMerge3DHeight() {
    PR_MERGE_3D_HEIGHT_CMD stCmd;
    PR_MERGE_3D_HEIGHT_RPY stRpy;

    cv::Mat matHeight;
    std::string strHeightResultPath = gstrWorkingFolder + "Height.yml";
    cv::FileStorage fs ( strHeightResultPath, cv::FileStorage::READ );
    cv::FileNode fileNode = fs["Height"];
    cv::read ( fileNode, matHeight, cv::Mat () );
    fs.release ();

    stCmd.vecMatHeight.push_back ( matHeight );
    cv::Mat matHeight1 = matHeight.clone();
    matHeight1 = matHeight1 + 0.1;
    stCmd.vecMatHeight.push_back ( matHeight1 );
    PR_Merge3DHeight ( &stCmd, &stRpy );
    std::cout << "PR_Merge3DHeight status " << ToInt32( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK != stRpy.enStatus )
        return;

    cv::Mat matHeightResultImg = _drawHeightGrid ( stRpy.matHeight, 10, 10 );
    cv::imwrite ( gstrWorkingFolder + "PR_TestMerge3DHeight_HeightGridImg.png", matHeightResultImg );
}

void TestCalc3DHeightDiff(const cv::Mat &matHeight) {
    PR_CALC_3D_HEIGHT_DIFF_CMD stCmd;
    PR_CALC_3D_HEIGHT_DIFF_RPY stRpy;
    stCmd.matHeight = matHeight;
    //stCmd.vecRectBases.push_back ( cv::Rect (761, 1783, 161, 113 ) );
    //stCmd.vecRectBases.push_back ( cv::Rect (539, 1370, 71, 32 ) );
    //stCmd.rectROI = cv::Rect(675, 1637, 103, 78 );
    stCmd.vecRectBases.push_back ( cv::Rect (550, 787, 95, 184 ) );
    stCmd.rectROI = cv::Rect ( 709, 852, 71, 69 );
    PR_Calc3DHeightDiff ( &stCmd, &stRpy );
    std::cout << "PR_Calc3DHeightDiff status " << ToInt32( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK == stRpy.enStatus ) {
        std::cout << "Height Diff Result " << stRpy.fHeightDiff << std::endl;
    }
}

void TestCalcMTF() {
    const int IMAGE_COUNT = 12;
    std::string strFolder = "./data/0831213010_25_Plane/";
    PR_CALC_MTF_CMD stCmd;
    PR_CALC_MTF_RPY stRpy;
    for ( int i = 1; i <= IMAGE_COUNT; ++ i ) {
        char chArrFileName[100];
        _snprintf( chArrFileName, sizeof (chArrFileName), "%02d.bmp", i );
        std::string strImageFile = strFolder + chArrFileName;
        cv::Mat mat = cv::imread ( strImageFile, cv::IMREAD_GRAYSCALE );
        stCmd.vecInputImgs.push_back ( mat );
    }
    stCmd.fMagnitudeOfDLP = 161;

    PR_CalcMTF ( &stCmd, &stRpy );
    std::cout << "PR_CalcMTF status " << ToInt32( stRpy.enStatus ) << std::endl;
}

void TestCalcPD() {
    const int IMAGE_COUNT = 12;
    //std::string strFolder = "./data/0927225721_invert/";
    std::string strFolder = "./data/0715/0715184554_10ms_80_Plane1/";
    PR_CALC_PD_CMD stCmd;
    PR_CALC_PD_RPY stRpy;
    for ( int i = 1; i <= IMAGE_COUNT; ++ i ) {
        char chArrFileName[100];
        _snprintf( chArrFileName, sizeof (chArrFileName), "%02d.bmp", i );
        std::string strImageFile = strFolder + chArrFileName;
        cv::Mat mat = cv::imread ( strImageFile, cv::IMREAD_GRAYSCALE );
        stCmd.vecInputImgs.push_back ( mat );
    }
    stCmd.fMagnitudeOfDLP = 161;

    PR_CalcPD ( &stCmd, &stRpy );
    std::cout << "PR_CalcPD status " << ToInt32( stRpy.enStatus ) << std::endl;
    if ( VisionStatus::OK == stRpy.enStatus ) {
        cv::imwrite("./data/CaptureRegionImg.png", stRpy.matCaptureRegionImg );
    }
}