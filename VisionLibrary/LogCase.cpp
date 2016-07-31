#include "LogCase.h"
#include <vector>
#include <sstream>
//#include <ctime>
#include "opencv2/highgui.hpp"
#include "boost/format.hpp"
#include "boost/filesystem.hpp"
#include "StopWatch.h"
#include "SimpleIni.h"
#include "VisionAlgorithm.h"

namespace bfs = boost::filesystem;

namespace AOI
{
namespace Vision
{

//Define local functions
namespace
{
    StringVector split(const String &s, char delim)
    {
        StringVector elems;
        std::stringstream ss(s);
        String item;
        while (getline(ss, item, delim)) {
            elems.push_back(item);
        }
        return elems;
    }
}

String LogCase::_formatCoordinate(const Point2f &pt)
{
    char chArray[200];
    _snprintf( chArray, sizeof ( chArray), "%.4f, %.4f", pt.x, pt.y );
    return String (chArray);
}

Point2f LogCase::_parseCoordinate(const String &strCoordinate)
{
    Point2f pt(0.f, 0.f);
    StringVector vecStrCoordinate = split(strCoordinate, ',');
    if ( vecStrCoordinate.size() == 2 ) {
        pt.x = std::stof ( vecStrCoordinate[0] );
        pt.y = std::stof ( vecStrCoordinate[1] );
    }
    return pt;
}

String LogCase::_formatRect(const Rect2f &pt)
{
    char chArray[200];
    _snprintf(chArray, sizeof(chArray), "%.4f, %.4f, %.4f, %.4f", pt.tl().x, pt.tl().y, pt.br().x, pt.br().y);
    return String(chArray);
}

Rect2f LogCase::_parseRect(const String &strCoordinate)
{
    Point2f pt1(0.f, 0.f), pt2(0.f, 0.f);
    StringVector vecStrCoordinate = split(strCoordinate, ',');
    if ( vecStrCoordinate.size() == 4 ) {
        pt1.x = std::stof ( vecStrCoordinate[0] );
        pt1.y = std::stof ( vecStrCoordinate[1] );
        pt2.x = std::stof ( vecStrCoordinate[2] );
        pt2.y = std::stof ( vecStrCoordinate[3] );
    }
    return Rect2f(pt1, pt2);
}

String LogCase::_generateLogCaseName(const String &strFolderPrefix)
{
    auto timeT = std::time(nullptr);
    auto stTM = std::localtime(&timeT);
    CStopWatch stopWatch;
    auto nMilliseconds = stopWatch.AbsNow () - timeT * 1000;
    String strFmt = "_%04s_%02s_%02s_%02s_%02s_%02s_%03s";
    String strTime = (boost::format(strFmt) % (stTM->tm_year + 1900) % (stTM->tm_mon + 1) % stTM->tm_mday % stTM->tm_hour % stTM->tm_min % stTM->tm_sec % nMilliseconds).str();
    String strLogCasePath = _strLogCasePath + strFolderPrefix + strTime + "\\";
    bfs::path dir(strLogCasePath);
    bfs::create_directories(dir);
    return strLogCasePath;
}

const String LogCaseLrnTmpl::FOLDER_PREFIX = "LrnTmpl";

LogCaseLrnTmpl::LogCaseLrnTmpl(const String &strPath, bool bReplay) : LogCase(strPath)
{
    if ( bReplay )
        _strLogCasePath = strPath;
    else
        _strLogCasePath = _generateLogCaseName( FOLDER_PREFIX );
}

VisionStatus LogCaseLrnTmpl::WriteCmd(PR_LRN_TMPL_CMD *pLrnTmplCmd)
{
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile( cmdRpyFilePath.c_str() );
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyAlgorithm.c_str(), static_cast<long>(pLrnTmplCmd->enAlgorithm) );
    ini.SetValue(_CMD_SECTION.c_str(), _strKeyLrnWindow.c_str(), _formatRect(pLrnTmplCmd->rectLrn).c_str() );
    ini.SaveFile( cmdRpyFilePath.c_str() );
    cv::imwrite( _strLogCasePath + "image.jpg", pLrnTmplCmd->mat );
    if ( ! pLrnTmplCmd->mask.empty() )
        cv::imwrite( _strLogCasePath + "mask.jpg", pLrnTmplCmd->mask );
    return VisionStatus::OK;
}

VisionStatus LogCaseLrnTmpl::WriteRpy(PR_LRN_TMPL_RPY *pLrnTmplRpy)
{
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile( cmdRpyFilePath.c_str() );
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), static_cast<long>(pLrnTmplRpy->nStatus) );
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyCenterPos.c_str(), _formatCoordinate (pLrnTmplRpy->ptCenter ).c_str() );

    ini.SaveFile( cmdRpyFilePath.c_str() );
    return VisionStatus::OK;
}

VisionStatus LogCaseLrnTmpl::RunLogCase()
{
    PR_LRN_TMPL_CMD stLrnTmplCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile( cmdRpyFilePath.c_str() );
    stLrnTmplCmd.enAlgorithm = static_cast<PR_ALIGN_ALGORITHM> ( ini.GetLongValue ( _CMD_SECTION.c_str(), _strKeyAlgorithm.c_str(), ToInt32(PR_ALIGN_ALGORITHM::SURF) ) );
    stLrnTmplCmd.rectLrn = _parseRect ( ini.GetValue(_CMD_SECTION.c_str(), _strKeyLrnWindow.c_str(), "" ) );
    stLrnTmplCmd.mat = cv::imread( _strLogCasePath + "image.jpg" );
    stLrnTmplCmd.mask = cv::imread( _strLogCasePath + "mask.jpg" );

    PR_LRN_TMPL_RPY stLrnTmplRpy;

    std::shared_ptr<VisionAlgorithm> pVA = VisionAlgorithm::create();
    enStatus = pVA->lrnTmpl ( &stLrnTmplCmd, &stLrnTmplRpy, true );

    WriteRpy( &stLrnTmplRpy );
    return enStatus;
}

const String LogCaseLrnDevice::FOLDER_PREFIX = "LrnDevice";

}
}