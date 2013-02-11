#include <iostream>
#include <vector>
#include <sstream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/gpu/gpu.hpp"

#include "imageLib.h"
#include "flowIO.h"
#include "colorcode.h"

using namespace std;
using namespace cv;
using namespace cv::gpu;

/* \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ */

template <typename T> inline T clamp (T x, T a, T b)
{
    return ((x) > (a) ? ((x) < (b) ? (x) : (b)) : (a));
}

/* \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ */

template <typename T> inline T mapValue(T x, T a, T b, T c, T d)
{
    x = clamp(x, a, b);
    return c + (d - c) * (x - a) / (b - a);
}

Mat_<Vec3b> MotionToColor(CFloatImage &motim, float maxmotion);
void flowToImage(const Mat& flow, CFloatImage& img);
void getFlowField(const Mat& u, const Mat& v, Mat& flowField);

Mat_<Vec2f> getFlowField(const Mat& u, const Mat& v)
{
    Mat_<Vec2f> img(u.rows, u.cols, Vec2f(255.f,255.f));
    for (int y = 0; y < u.rows; ++y)
    {
        for (int x = 0; x < u.cols; ++x)
        {
            img(y, x)[0] = -v.at<float>(y, x);
            img(y, x)[1] = u.at<float>(y, x);

            if(u.at<float>(y,x) > 0.1) 
            {
              std::cout << img(y, x)[0] << " " << img(y, x)[1] << std::endl;
            }
        }
    }
    return img; 
    //std::cout << img.rows << " " << img.cols << std::endl;
    //std::cerr << "finished getFlowFiled" << std::endl;
}
/* \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ */

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

/* \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ */

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


template <typename T>
inline T mapVal(T x, T a, T b, T c, T d)
{
    x = ::max(::min(x, b), a);
    return c + (d-c) * (x-a) / (b-a);
}

static void colorizeFlow(const Mat &u, const Mat &v, Mat &dst)
{
    double uMin, uMax;
    minMaxLoc(u, &uMin, &uMax, 0, 0);
    double vMin, vMax;
    minMaxLoc(v, &vMin, &vMax, 0, 0);
    uMin = ::abs(uMin); uMax = ::abs(uMax);
    vMin = ::abs(vMin); vMax = ::abs(vMax);
    float dMax = static_cast<float>(::max(::max(uMin, uMax), ::max(vMin, vMax)));

    dst.create(u.size(), CV_8UC3);
    for (int y = 0; y < u.rows; ++y)
    {
        for (int x = 0; x < u.cols; ++x)
        {
            dst.at<uchar>(y,3*x) = 0;
            dst.at<uchar>(y,3*x+1) = (uchar)mapVal(-v.at<float>(y,x), -dMax, dMax, 0.f, 255.f);
            dst.at<uchar>(y,3*x+2) = (uchar)mapVal(u.at<float>(y,x), -dMax, dMax, 0.f, 255.f);
        }
    }
}

int main(int argc, char **argv)
{
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    namedWindow("current_frame", 2);
    CommandLineParser cmd(argc, argv,
            "{ l | left | | specify left image }"
            "{ r | right | | specify right image }"
            "{ h | help | false | print help message }");

    if (cmd.get<bool>("help"))
    {
        cout << "Farneback's optical flow sample.\n\n"
             << "Usage: farneback_optical_flow_gpu [arguments]\n\n"
             << "Arguments:\n";
        cmd.printParams();
        return 0;
    }

    /*string pathL = cmd.get<string>("left");
    string pathR = cmd.get<string>("right");
    if (pathL.empty()) cout << "Specify left image path\n";
    if (pathR.empty()) cout << "Specify right image path\n";
    if (pathL.empty() || pathR.empty()) return -1;

    Mat frameL = imread(pathL, IMREAD_GRAYSCALE);
    Mat frameR = imread(pathR, IMREAD_GRAYSCALE);
    */
    Mat frameL;
    Mat frameR;
    
    cap >> frameL;
    while(true)
    {
    cap >> frameR;

    //if (frameL.empty()) cout << "Can't open '" << pathL << "'\n";
    //if (frameR.empty()) cout << "Can't open '" << pathR << "'\n";
    //if (frameL.empty() || frameR.empty()) return -1;

    //frameL.convertTo(frameL, CV_32F, 1.0 / 255.0);
    //frameR.convertTo(frameR, CV_32F, 1.0 / 255.0);
    //frameL.convertTo(frameL, CV_8U);
    //frameL.convertTo(frameL, CV_8U);

    Mat frameLGray;
    Mat frameRGray;

    cvtColor(frameL, frameLGray, COLOR_BGR2GRAY);
    cvtColor(frameR, frameRGray, COLOR_BGR2GRAY);

    GpuMat d_frameL(frameLGray), d_frameR(frameRGray);
    imshow("current_frame", frameRGray);
    GpuMat d_flowx, d_flowy;
    FarnebackOpticalFlow d_calc;
    Mat flowxy, flowx, flowy, image;

    bool running = true, gpuMode = true;
    int64 t, t0=0, t1=1, tc0, tc1;

    cout << "Use 'm' for CPU/GPU toggling\n";

    t = getTickCount();

        tc0 = getTickCount();
        d_calc(d_frameL, d_frameR, d_flowx, d_flowy);
        tc1 = getTickCount();
        d_flowx.download(flowx);
        d_flowy.download(flowy);

        //Mat flow;
        Mat_<Vec2f> flow = getFlowField(flowx, flowy);
        std::cerr << " ** finished getFlowFiled" << std::endl;
        CFloatImage im, fband;

        std::cout << "XXXX\n"; 
        std::cout << flow.rows << " " << flow.cols << std::endl;
        flowToImage(flow, im);


        std::cerr << " ** flowToImage" << std::endl;
        Mat_<Vec3b> outim = MotionToColor(im, -1.0);  

        //colorizeFlow(flowx, flowy, image);

        stringstream s;
        //s << "mode: " << (gpuMode?"GPU":"CPU");
        putText(image, s.str(), Point(5, 25), FONT_HERSHEY_SIMPLEX, 1., Scalar(255,0,255), 2);

        s.str("");
        s << "opt. flow FPS: " << cvRound((getTickFrequency()/(tc1-tc0)));
        putText(image, s.str(), Point(5, 65), FONT_HERSHEY_SIMPLEX, 1., Scalar(255,0,255), 2);

        s.str("");
        s << "total FPS: " << cvRound((getTickFrequency()/(t1-t0)));
        putText(image, s.str(), Point(5, 105), FONT_HERSHEY_SIMPLEX, 1., Scalar(255,0,255), 2);

        //imshow("flow", image);
        imshow("flow", outim);

        char ch = (char)waitKey(3);
        /*if (ch == 27)
            running = false;
        else if (ch == 'm' || ch == 'M')
            gpuMode = !gpuMode;
            */

        t0 = t;
        t1 = getTickCount();
        frameL = frameR;
        if(waitKey(30) >= 0) break;
    }

    return 0;
}
