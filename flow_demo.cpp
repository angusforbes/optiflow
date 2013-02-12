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

int main(int argc, char*argv[])
{
    VideoCapture cap(0); // open the default camera
    //VideoCapture cap;
    //cap.open("/Users/jguan/data/psi/movies/map.avi");
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
        //resize(next_frame, next_frame_small, Size(), 0.25, 0.25, INTER_NEAREST);

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
        std::cout << cur_frame_small.at<Vec2f>(10, 11) << std::endl;
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
            writeOpticalFlowToFile(flow, flow_str);
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
