// flowIO.h
#ifndef FLOW_IO_H
#define FLOW_IO_H

#include <opencv2/opencv.hpp>

// the "official" threshold - if the absolute value of either 
// flow component is greater, it's considered unknown
#define UNKNOWN_FLOW_THRESH 1e9

// value to use to represent unknown flow
#define UNKNOWN_FLOW 1e10

// return whether flow vector is unknown
bool unknown_flow(float u, float v);
bool unknown_flow(float *f);

// read a flow file into 2-band image
void ReadFlowFile(CFloatImage& img, const char* filename);

// write a 2-band image into flow file 
void WriteFlowFile(CFloatImage img, const char* filename);

void flowToImage(const cv::Mat& flow, CFloatImage& img);

void writeOpticalFlowToFile(const cv::Mat& flow, const char* filename);

#endif
