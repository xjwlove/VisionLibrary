#ifndef _VISION_MATCH_TMPL_H_
#define _VISION_MATCH_TMPL_H_

#include "VisionHeader.h"

namespace AOI
{
namespace Vision
{

class MatchTmpl
{
public:
    MatchTmpl ();
    ~MatchTmpl ();
    static cv::Point matchByRecursionEdge(const cv::Mat &matInput, const cv::Mat &matTmpl, const cv::Mat &matMaskOfTmpl );
    static VisionStatus refineSrchTemplate(const cv::Mat &mat, const cv::Mat &matTmpl, PR_OBJECT_MOTION enMotion, cv::Point2f &ptResult, float &fRotation, float &fCorrelation);
    static VisionStatus matchTemplate(const cv::Mat &mat, const cv::Mat &matTmpl, bool bSubPixelRefine, PR_OBJECT_MOTION enMotion, cv::Point2f &ptResult, float &fRotation, float &fCorrelation, const cv::Mat &matMask = cv::Mat());
    static cv::Point myMatchTemplate(const cv::Mat &mat, const cv::Mat &matTmpl, const cv::Mat &matMask = cv::Mat());
    static cv::Point matchTemplateRecursive(const cv::Mat &matInput, const cv::Mat &matTmpl);
    static float calcCorrelation(const cv::Mat &matT, const cv::Mat &matI, const cv::Mat &matMaskT = cv::Mat(), const cv::Mat &matMaskI = cv::Mat());
};

}
}

#endif
