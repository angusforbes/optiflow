#include <stdlib.h>
//#include <opencv2/opencv.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"

#include <cstdio>
#include <iostream>

#include "imageLib.h"
#include "flowIO.h"
#include "colorcode.h"

using namespace cv;
using namespace std;
using namespace cv::gpu;

int verbose = 1;

void writeOpticalFlowToFile(const Mat& flow, FILE* file);

void MotionToColor(CFloatImage &motim, CByteImage &colim, float maxmotion);
Mat_<Vec3b> MotionToColor(CFloatImage &motim, float maxmotion);

void flowToImage(const Mat& flow, CFloatImage& img);

void getFlowField(const Mat& u, const Mat& v, Mat& flowField);


// binary file format for flow data specified here:
// http://vision.middlebury.edu/flow/data/
void writeOpticalFlowToFile(const Mat& flow, FILE* file) {
  int cols = flow.cols;
  int rows = flow.rows;

  fprintf(file, "PIEH");

  if (fwrite(&cols, sizeof(int), 1, file) != 1 ||
      fwrite(&rows, sizeof(int), 1, file) != 1) {
    printf("writeOpticalFlowToFile : problem writing header\n");
    exit(1);
  }

  for (int i= 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      Vec2f flow_at_point = flow.at<Vec2f>(i, j);

      if (fwrite(&(flow_at_point[0]), sizeof(float), 1, file) != 1 ||
          fwrite(&(flow_at_point[1]), sizeof(float), 1, file) != 1) {
        printf("writeOpticalFlowToFile : problem writing data\n");
        exit(1);
      }
    }
  }
}

/* \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ */

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

/* \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ */

void getFlowField(const Mat& u, const Mat& v, Mat& flowField)
{
    float maxDisplacement = 1.0f;

    for (int i = 0; i < u.rows; ++i)
    {
        const float* ptr_u = u.ptr<float>(i);
        const float* ptr_v = v.ptr<float>(i);

        for (int j = 0; j < u.cols; ++j)
        {
            float d = max(fabsf(ptr_u[j]), fabsf(ptr_v[j]));

            if (d > maxDisplacement)
                maxDisplacement = d;
        }
    }

    flowField.create(u.size(), CV_8UC4);

    for (int i = 0; i < flowField.rows; ++i)
    {
        const float* ptr_u = u.ptr<float>(i);
        const float* ptr_v = v.ptr<float>(i);


        Vec4b* row = flowField.ptr<Vec4b>(i);

        for (int j = 0; j < flowField.cols; ++j)
        {
            row[j][0] = 0;
            row[j][1] = static_cast<unsigned char> (mapValue (-ptr_v[j], -maxDisplacement, maxDisplacement, 0.0f, 255.0f));
            row[j][2] = static_cast<unsigned char> (mapValue ( ptr_u[j], -maxDisplacement, maxDisplacement, 0.0f, 255.0f));
            row[j][3] = 255;
        }
    }
}
int main(int argc, char*argv[])
{
    //VideoCapture cap(0); // open the default camera
    VideoCapture cap;
    cap.open("/Users/jguan/data/psi/movies/map.avi");
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    bool save = false;
    if(argc < 2)
    {
        std::cout << " Starting flow demo ... \n";
        std::cout << " You can start by " << argv[0] << " save " << 
                     " to save the frames to ./frames \n" << 
                     " flow (x, y) to ./flow/ \n " << 
                     " flow images to ./flow_image \n";
    }
    else if(argc == 2) 
    {
        std::cout << " Save info \n";
        save = true;
        system("mkdir -p  ./frames");
        system("mkdir -p  ./flows");
        system("mkdir -p  ./flow_images");
    }

    namedWindow("flow", 1);
    namedWindow("current_frame", 2);
    Mat cur_frame;
    Mat cur_frame_small;
    Mat next_frame;
    Mat next_frame_small;
    cap >> cur_frame;
    int frame = 1;
    while(true)
    {
        resize(cur_frame, cur_frame_small, Size(), 0.25, 0.25, INTER_NEAREST);
        if(save)
        {
            char frame_str[50];
            sprintf(frame_str, "./frames/%05d.jpg", frame);
            imwrite(frame_str, cur_frame); 
        }

        cap >> next_frame; // get a new frame from camera
        resize(next_frame, next_frame_small, Size(), 0.25, 0.25, INTER_NEAREST);

        // Scale the frames to doubles 
        cur_frame_small.convertTo(cur_frame_small, CV_32F, 1.0 / 255.0);
        cur_frame_small.convertTo(cur_frame_small, CV_32F, 1.0 / 255.0);

        Mat cur_frame_gray, next_frame_gray;
        cvtColor(cur_frame_small, cur_frame_gray, COLOR_BGR2GRAY);
        cvtColor(next_frame_small, next_frame_gray, COLOR_BGR2GRAY);

        //GPU
        cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());
        GpuMat d_frame_cur(cur_frame_gray);
        GpuMat d_frame_next(next_frame_gray);
        double alpha = 0.012;
        double gamma = 0.5;
        double scale_factor = 0.5;
        int inner_iter = 1;
        int outer_iter = 3;
        int sov_iter = 20;

        BroxOpticalFlow brox(alpha, gamma, scale_factor, inner_iter, outer_iter, sov_iter);

        GpuMat d_fu, d_fv;
        float start = (float)getTickCount();
        brox(d_frame_cur, d_frame_next, d_fu, d_fv);
        Mat flow;
        getFlowField(Mat(d_fu), Mat(d_fv), flow);
        
        printf("calcOpticalFlowSF : %lf sec\n", (getTickCount() - start) / getTickFrequency());
        std::cout << " Computed the flow" << std::endl;;

        if(save)
        {
            char flow_str[50];
            sprintf(flow_str, "./flows/%5d.flo", frame);
            FILE* file = fopen(flow_str, "wb");
            if (file == NULL) {
              printf("Unable to open file '%s' for writing\n", flow_str);
              exit(1);
            }
            writeOpticalFlowToFile(flow, file);
            fclose(file);
        }

	    CFloatImage im, fband;
        flowToImage(flow, im);

        float maxmotion = 5;
        Mat_<Vec3b> outim = MotionToColor(im, maxmotion);  

        imshow("flow", outim);
        imshow("current_frame", next_frame_small);
        if(save)
        {
            char img_str[50];
            sprintf(img_str, "./flow_images/%5d.jpg", frame);
            imwrite(img_str, outim); 
        }
        frame++;
        cur_frame = next_frame;
        if(waitKey(30) >= 0) break;
    }

    return 0;
}

