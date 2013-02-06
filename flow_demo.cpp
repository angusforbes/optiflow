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

void writeOpticalFlowToFile(const Mat& flow, FILE* file);

void MotionToColor(CFloatImage &motim, CByteImage &colim, float maxmotion);
Mat_<Vec3b> MotionToColor(CFloatImage &motim, float maxmotion);

void flowToImage(const Mat& flow, CFloatImage& img);

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
    VideoCapture cap(0); // open the default camera
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
        resize(cur_frame, cur_frame_small, Size(), 0.10, 0.10, INTER_NEAREST);
        if(save)
        {
            char frame_str[50];
            sprintf(frame_str, "./frames/%05d.jpg", frame);
            imwrite(frame_str, cur_frame); 
        }

        cap >> next_frame; // get a new frame from camera
        resize(next_frame, next_frame_small, Size(), 0.10, 0.10, INTER_NEAREST);

        Mat flow;

        // Flow parameters
        float start = (float)getTickCount();
        //int layers = 3;
        int layers = 1; //AGF
        //int averaging_block_size = 5;
        int averaging_block_size = 1; //AGF
        int max_flow = 4;
        //double sigma_dist = 4.1;
        double sigma_dist = 5.5;
        //double sigma_color = 25.5;
        double sigma_color = 0.08;
        //int postprocess_window = 20;
        int postprocess_window = 10; //AGF
        double sigma_dist_fix = 55.0;
        double sigma_color_fix = 25.5;
        double occ_thr = 0.35;
        //int upscale_averaging_radius = 18;
        int upscale_averaging_radius = 18; //AGF
        double upscale_sigma_dist = 55.0;
        double upscale_sigma_color = 25.5;
        //double speed_up_thr = 10;
       // double speed_up_thr = 5.0;
        double speed_up_thr = 5.0;
        calcOpticalFlowSF(cur_frame_small, next_frame_small,
                         flow,
                         layers, averaging_block_size, max_flow, sigma_dist, sigma_color,
                         postprocess_window, sigma_dist_fix, sigma_color_fix, occ_thr, 
                         upscale_averaging_radius, upscale_sigma_dist, upscale_sigma_color, 
                         speed_up_thr);
        //calcOpticalFlowSF(cur_frame_small, next_frame_small,
        //                 flow,
        //                 3, 5, 10);
        
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
