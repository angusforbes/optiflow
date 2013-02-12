#ifndef COLORCODE_H
#define COLORCODE_H
#include <opencv2/opencv.hpp>
#include "imageLib.h"

void computeColor(float fx, float fy, uchar *pix);

cv::Mat_<cv::Vec3b> MotionToColor(CFloatImage &motim, float maxmotion);

#endif
