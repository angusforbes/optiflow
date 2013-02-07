// Example showing how to read and write images
//#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
//#include <opencv/cvaux.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/highgui.hpp"

#include <cstdio>
#include <iostream>
#include <stdlib.h>

using namespace cv;
using namespace std;

/*
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
*/

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
        resize(cur_frame, cur_frame_small, Size(), 0.50, 0.50, INTER_NEAREST);
        
        cap >> next_frame; // get a new frame from camera
        resize(next_frame, next_frame_small, Size(), 0.50, 0.50, INTER_NEAREST);
        
        imshow("current_frame", next_frame_small);
        
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
        
         imshow("flow", cur_frame_small);
        
        frame++;
        cur_frame = next_frame;
    }
    
}

/*
int main(int argc, char** argv)
{
    IplImage * pInpImg = 0;
 
    // Load an image from file - change this based on your image name
    pInpImg = cvLoadImage("/Users/angus.forbes/Dropbox/XCodeProjects/optiflow/optiflowCV/test.jpg", CV_LOAD_IMAGE_UNCHANGED);
    if(!pInpImg)
    {
        fprintf(stderr, "failed to load input image\n");
        return -1;
    }
    
    // Write the image to a file with a different name,
    // using a different image format -- .png instead of .jpg
    if( !cvSaveImage("/Users/angus.forbes/Dropbox/XCodeProjects/optiflow/optiflowCV/my_image_copy.png", pInpImg) )
    {
        fprintf(stderr, "failed to write image file\n");
    }
    
    // Remember to free image memory after using it!
    cvReleaseImage(&pInpImg);
    
    return 0;
}
*/