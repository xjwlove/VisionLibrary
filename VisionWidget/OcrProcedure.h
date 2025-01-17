#pragma once
#include "Procedure.h"

class OcrProcedure :
    public Procedure
{
public:
    OcrProcedure(VisionView *pVisionView);
    ~OcrProcedure();
    virtual int run(const std::string &imagePath) override;
    std::string  getResult() const;
    void setTextDirection(int nTextDirection);
private:
    cv::Rect                        _rectSrchWindow;
    int                             _nTextDirection;
    int ocr(const std::string &imagePath);
    std::string                     _strResult;
};