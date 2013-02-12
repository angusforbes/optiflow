#include <iostream>
#include <iomanip>
#include <string>
#include <cstdio>

#include "opencv2/core/core.hpp"
#include "opencv2/core/opengl_interop.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"

#include "imageLib.h"
#include "flowIO.h"
#include "colorcode.h"

using namespace std;
using namespace cv;
using namespace cv::gpu;
int verbose = 1;
void getFlowField(const Mat& u, const Mat& v, Mat& flowField);
void writeOpticalFlowToFile(const Mat& flow, FILE* file);

int main(int argc, const char* argv[])
{
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;
    try
    {
        const char* keys =
           "{ h  | help      | false | print help message }"
           "{ l  | left      |       | specify left image }"
           "{ r  | right     |       | specify right image }"
           "{ s  | scale     | 0.8   | set pyramid scale factor }"
           "{ a  | alpha     | 0.197 | set alpha }"
           "{ g  | gamma     | 50.0  | set gamma }"
           "{ i  | inner     | 10    | set number of inner iterations }"
           "{ o  | outer     | 77    | set number of outer iterations }"
           "{ si | solver    | 10    | set number of basic solver iterations }"
           "{ t  | time_step | 0.1   | set frame interpolation time step }";

        CommandLineParser cmd(argc, argv, keys);

        if (cmd.get<bool>("help"))
        {
            cout << "Usage: brox_optical_flow [options]" << endl;
            cout << "Avaible options:" << endl;
            cmd.printParams();
            return 0;
        }

        //string frame0Name = cmd.get<string>("left");
        //string frame1Name = cmd.get<string>("right");
        float scale = cmd.get<float>("scale");
        float alpha = cmd.get<float>("alpha");
        float gamma = cmd.get<float>("gamma");
        int inner_iterations = cmd.get<int>("inner");
        int outer_iterations = cmd.get<int>("outer");
        int solver_iterations = cmd.get<int>("solver");
        float timeStep = cmd.get<float>("time_step");

        //Mat frame0Color = imread(frame0Name);
        //Mat frame1Color = imread(frame1Name);
        Mat frame0Color;
        Mat frame1Color;

        namedWindow("flow", 1);
        namedWindow("current_frame", 2);

        cap >> frame0Color;
        while(true)
        {
            cap >> frame1Color;
            if (frame0Color.empty() || frame1Color.empty())
            {
                cout << "Can't load input images" << endl;
                return -1;
            }

            cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());

            frame0Color.convertTo(frame0Color, CV_32F, 1.0 / 255.0);
            frame1Color.convertTo(frame1Color, CV_32F, 1.0 / 255.0);

            Mat frame0Gray, frame1Gray;

            cvtColor(frame0Color, frame0Gray, COLOR_BGR2GRAY);
            cvtColor(frame1Color, frame1Gray, COLOR_BGR2GRAY);

            GpuMat d_frame0(frame0Gray);
            GpuMat d_frame1(frame1Gray);

            cout << "Estimating optical flow" << endl;

            BroxOpticalFlow d_flow(alpha, gamma, scale, inner_iterations, outer_iterations, solver_iterations);

            cout << "\tForward..." << endl;

            GpuMat d_fu, d_fv;

            d_flow(d_frame0, d_frame1, d_fu, d_fv);
                    
            Mat flow;
            getFlowField(Mat(d_fu), Mat(d_fv), flow);
            /*FILE* file = fopen("./flow.flo", "wb");
            if (file == NULL) {
              printf("Unable to open file ./flow.flo for writing\n");
              exit(1);
            }
            writeOpticalFlowToFile(flow, file);
            */
            CFloatImage im, fband;
            flowToImage(flow, im);

            float maxmotion = -1.0;
            Mat_<Vec3b> outim = MotionToColor(im, maxmotion);  

            imshow("flow", outim);
            imshow("current_frame", frame1Color);
            frame0Color = frame1Color;

            if(waitKey(30) >= 0) break;
        }

    }
    catch (const exception& ex)
    {
        cerr << ex.what() << endl;
        return -1;
    }
    catch (...)
    {
        cerr << "Unknow error" << endl;
        return -1;
    }

    return 0;
}

template <typename T> inline T clamp (T x, T a, T b)
{
    return ((x) > (a) ? ((x) < (b) ? (x) : (b)) : (a));
}

template <typename T> inline T mapValue(T x, T a, T b, T c, T d)
{
    x = clamp(x, a, b);
    return c + (d - c) * (x - a) / (b - a);
}

void getFlowField(const Mat& u, const Mat& v, Mat& flowField)
{
    /*float maxDisplacement = 1.0f;

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
    */
    int cols = u.cols;
    int rows = u.rows;
    flowField.create(rows, cols, CV_32FC2);
    for(int y = 0; y < rows; y++)
    {
        for(int x = 0; x < cols; x++)
        {
            flowField.at<Vec2f>(y, x)[0] = u.at<float>(y, x);
            flowField.at<Vec2f>(y, x)[1] = v.at<float>(y, x);
        }
    }
}

/* \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ */

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

