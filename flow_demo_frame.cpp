#include <stdlib.h>
//#include <opencv2/opencv.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <cstdio>
#include <iostream>

#include "imageLib.h"
#include "flowIO.h"
#include "colorcode.h"

using namespace cv;
using namespace std;

int verbose = 1;
void MotionToColor(CFloatImage &motim, CByteImage &colim, float maxmotion);
Mat_<Vec3b> MotionToColor(CFloatImage &motim, float maxmotion);

void flowToImage(const Mat& flow, CFloatImage& img);

void MotionToColor(CFloatImage &motim, CByteImage &colim, float maxmotion)
{
    CShape sh = motim.Shape();
    int width = sh.width, height = sh.height;
    colim.ReAllocate(CShape(width, height, 3));
    int x, y;
    // determine motion range:
    float maxx = -999, maxy = -999;
    float minx =  999, miny =  999;
    float maxrad = -1;
    for (y = 0; y < height; y++) {
	for (x = 0; x < width; x++) {
	    float fx = motim.Pixel(x, y, 0);
	    float fy = motim.Pixel(x, y, 1);
	    if (unknown_flow(fx, fy))
		continue;
	    maxx = __max(maxx, fx);
	    maxy = __max(maxy, fy);
	    minx = __min(minx, fx);
	    miny = __min(miny, fy);
	    float rad = sqrt(fx * fx + fy * fy);
	    maxrad = __max(maxrad, rad);
        }
    }
    printf("max motion: %.4f  motion range: u = %.3f .. %.3f;  v = %.3f .. %.3f\n",
	   maxrad, minx, maxx, miny, maxy);


    if (maxmotion > 0) // i.e., specified on commandline
	maxrad = maxmotion;

    if (maxrad == 0) // if flow == 0 everywhere
	maxrad = 1;

    if (verbose)
	fprintf(stderr, "normalizing by %g\n", maxrad);

    for (y = 0; y < height; y++) 
    {
        for (x = 0; x < width; x++) 
        {
            float fx = motim.Pixel(x, y, 0);
            float fy = motim.Pixel(x, y, 1);
            uchar *pix = &colim.Pixel(x, y, 0);
            if (unknown_flow(fx, fy)) 
            {
                pix[0] = pix[1] = pix[2] = 0;
            } 
            else 
            {
                computeColor(fx/maxrad, fy/maxrad, pix);
            }
        }
    }
}

Mat_<Vec3b> MotionToColor(CFloatImage &motim, float maxmotion)
{
    CShape sh = motim.Shape();
    int width = sh.width, height = sh.height;
    //Mat_<Vec3b> img(width, height, Vec3b(0,255,0));
    Mat_<Vec3b> img(height, width, Vec3b(0,255,0));
    int x, y;
    // determine motion range:
    float maxx = -999, maxy = -999;
    float minx =  999, miny =  999;
    float maxrad = -1;
    for (y = 0; y < height; y++) {
	for (x = 0; x < width; x++) {
	    float fx = motim.Pixel(x, y, 0);
	    float fy = motim.Pixel(x, y, 1);
	    if (unknown_flow(fx, fy))
		continue;
	    maxx = __max(maxx, fx);
	    maxy = __max(maxy, fy);
	    minx = __min(minx, fx);
	    miny = __min(miny, fy);
	    float rad = sqrt(fx * fx + fy * fy);
	    maxrad = __max(maxrad, rad);
        }
    }
    printf("max motion: %.4f  motion range: u = %.3f .. %.3f;  v = %.3f .. %.3f\n",
	   maxrad, minx, maxx, miny, maxy);


    if (maxmotion > 0) // i.e., specified on commandline
	maxrad = maxmotion;

    if (maxrad == 0) // if flow == 0 everywhere
	maxrad = 1;

    if (verbose)
	fprintf(stderr, "normalizing by %g\n", maxrad);

    for (y = 0; y < height; y++) 
    {
        for (x = 0; x < width; x++) 
        {
            float fx = motim.Pixel(x, y, 0);
            float fy = motim.Pixel(x, y, 1);
            if (unknown_flow(fx, fy)) 
            {
                img(y, x)[0] = img(y, x)[1] = img(y, x)[2] = 0.0;
            } 
            else 
            {
                uchar pix[3] = {0.0, 0.0 ,0.0}; 
                computeColor(fx/maxrad, fy/maxrad, pix);
                img(y, x)[0] = pix[0];
                img(y, x)[1] = pix[1];
                img(y, x)[2] = pix[2];
            }
        }
    }
    return img;
}

void flowToImage(const Mat& flow, CFloatImage& img)
{
    int cols = flow.cols;
    int rows = flow.rows;
    int nBands = 2;
    CShape sh(cols, rows, nBands);
    img.ReAllocate(sh);
    for (int i = 0; i < rows; i++) 
    {
        for (int j = 0; j < cols; j++) 
        {
            Vec2f flow_at_point = flow.at<Vec2f>(i, j);
            img.Pixel(j, i, 0) = flow_at_point[0];
            img.Pixel(j, i, 1) = flow_at_point[1];
        }
    }
}

int main(int argc, char*argv[])
{
    //VideoCapture cap(0); // open the default camera
    //if(!cap.isOpened())  // check if we succeeded
    //    return -1;

    //Mat edges;
    //namedWindow("flow",1);
    //namedWindow("current_frame",2);
    Mat cur_frame = imread(argv[1]);
    Mat next_frame = imread(argv[2]);
    //Mat cur_frame;
    Mat cur_frame_small;
    //Mat next_frame;
    Mat next_frame_small;
    char* outname = argv[3];
    //cap >> cur_frame;
    resize(cur_frame, cur_frame_small, Size(), 0.25, 0.25, INTER_NEAREST);
    //while(true)
    {
        //imwrite("frame_1.jpg", cur_frame); 
        //waitKey(2);
        //cap >> next_frame; // get a new frame from camera
        resize(next_frame, next_frame_small, Size(), 0.25, 0.25, INTER_NEAREST);
        //imwrite("frame_2.jpg", next_frame); 
        Mat flow;

        float start = (float)getTickCount();
        //calcOpticalFlowSF(cur_frame_small, next_frame_small,
        //                 flow,
        //                 3, 2, 4, 4.1, 25.5, 18, 55.0, 25.5, 0.35, 18, 55.0, 25.5, 10);
        //calcOpticalFlowSF(cur_frame_small, next_frame_small,
        //                 flow,
        //                 3, 5, 10);
        //calcOpticalFlowSF(cur_frame_small, next_frame_small, flow, 3, 5, 5);
        calcOpticalFlowFarneback(cur_frame_small, next_frame_small, flow, 0.5, 1, 4, 10, 5, 1.1, OPTFLOW_FARNEBACK_GAUSSIAN);
        printf("calcOpticalFlowSF : %lf sec\n", (getTickCount() - start) / getTickFrequency());
        std::cout << " Computed the flow" << std::endl;;

	    CFloatImage im, fband;
        flowToImage(flow, im);

        float maxmotion = -1;
        Mat_<Vec3b> outim = MotionToColor(im, maxmotion);  
	    //WriteImageVerb(outim, outname, verbose);
	    /*CByteImage outByteim;
	    CShape sh = im.Shape();
	    sh.nBands = 3;
	    outByteim.ReAllocate(sh);
	    outByteim.ClearPixels();
	    MotionToColor(im, outByteim, maxmotion);
	    WriteImageVerb(outByteim, outname, verbose);
        */

        //imshow("flow", outim);
        //imshow("current_frame", next_frame);
        imwrite("out.jpg", outim); 
        //cur_frame = next_frame;
        //if(waitKey(30) >= 0) break;
    }

    return 0;
}
