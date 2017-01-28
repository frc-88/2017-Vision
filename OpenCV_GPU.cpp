#include "networktables/NetworkTable.h"
#include "ntcore.h"
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <string>


using namespace cv;
using std::shared_ptr;
Mat frame;
Mat frame2;
gpu::GpuMat gpuFrame;
int setHeight = 94;
int setDist = 37;
float maxDist;
float oldDist;
int distBetween = 6;
double hFOV = 62;


NetworkTable *table;

Mat visionProcessing (Mat gpuMAT)
{
    gpu::GpuMat gpuFrame;
    gpuFrame.upload(gpuMAT);
    gpu::GpuMat normalizeInput = gpuFrame;
    gpu::GpuMat normalizeOutput;
    int normalizeType = NORM_INF;
    double normalizeAlpha = 150.0;
    double normalizeBeta = 200.0;
    gpu::normalize(normalizeInput, normalizeOutput, normalizeAlpha, normalizeBeta, normalizeType);
    //Step HSV_Threshold0:
    //input
    gpu::GpuMat gpuhsvThresholdInput;
    Mat hsvThresholdOutput;
    gpu::cvtColor(normalizeOutput, gpuhsvThresholdInput, COLOR_BGR2HSV);
    double hue[] = {60, 95.0};
    double sat[] = {50, 250};
    double val[] = {50, 250};
    Mat hsvThresholdInput(gpuhsvThresholdInput);
    inRange(hsvThresholdInput,Scalar(30, 55, 55), Scalar(75, 255, 247), hsvThresholdOutput);
    //gpu::GpuMat drawing = hsvThresholdOutput;
    //Mat findContoursInput = hsvThresholdOutput;
    Mat findContoursInput;
    int morph_size = 3;
    Mat element = getStructuringElement(MORPH_RECT, Size(2*morph_size+1, 2*morph_size+1), Point(morph_size, morph_size));
    //morphologyEx(hsvThresholdOutput, opening, MORPH_OPEN, element);
    morphologyEx(hsvThresholdOutput, findContoursInput, MORPH_CLOSE, element);
    return findContoursInput;
}

void setDriver(int b, int c, int e, int sh, int sa){
    string cmd = "driversettings ";
    if(b!=-1){
        cmd = cmd + "-b=" + std::to_string(b) + " ";
    }
    if(c!=-1){
        cmd = cmd + "-c=" + std::to_string(c) + " ";
    }
    if(e!=-1){
        cmd = cmd + "-e=" + std::to_string(e) + " ";
    }
    if(sa!=-1){
        cmd = cmd + "-s=" + std::to_string(sa) + " ";
    }
    if(sh!=-1){
        cmd = cmd + "-sh=" + std::to_string(sh) + " ";
    }

    //std::cout<<"Running: "<<cmd;

    const char *cmdca = cmd.c_str();

    system(cmdca);


}

int main(int, char**)
{
    bool verbose = false;
    int brightness = -1;
    int contrast = -1;
    int exposure = -1;
    int sharpness = -1;
    int saturation = -1;
    setDriver(brightness, contrast, exposure, sharpness, saturation);
    NetworkTable::SetClientMode();
    NetworkTable::SetIPAddress("10.0.88.107");
    NetworkTable::Initialize();
    shared_ptr<NetworkTable> table = NetworkTable::GetTable("imfeelinglucky");
    //table = NetworkTable::GetTable("imfeelinglucky");
    VideoCapture cap(-1); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;
    cap.set(CV_CAP_PROP_EXPOSURE, -200);
    cap.set(CV_CAP_PROP_FRAME_WIDTH,320);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT,240);
    //VideoCapture cap2(1);
    Mat edges;
    namedWindow("raw", CV_WINDOW_AUTOSIZE);
    namedWindow("edges",CV_WINDOW_AUTOSIZE);
    for(;;)
    {
        cap >> frame; // get a new frame from camera
      //  cap2 >> frame2;
        if (frame.empty())
            break;
        Mat findContoursInput;
        findContoursInput = visionProcessing(frame);
        bool externalOnly = false;
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        contours.clear();
        int mode = externalOnly ? RETR_EXTERNAL : RETR_LIST;
        int method = CHAIN_APPROX_SIMPLE;
        findContours(findContoursInput, contours, hierarchy, mode, method);
        float w_threshold = 250;
        float wl_threshold = 30;
        float h_threshold = 250;
        float hl_threshold = 5;
        vector<int> selected;
        vector<double> centerX;
        vector<double> centerY;
        vector<double> allHeight;
        vector<double> allWidth;
        Mat imageFinal(findContoursInput);
        int k = 0;
        for (int i = 0; i < contours.size(); i++)
        {
            if(verbose){
            std::cout<<contours.size()<<"\n";
            }
            Rect R = boundingRect(contours[i]);

            // filter contours according to their bounding box
            if (R.width < w_threshold && R.height < h_threshold && R.width > wl_threshold && R.height > hl_threshold)
            {
                selected.push_back(i);
                allHeight.push_back((double)R.height);
                allWidth.push_back((double)R.width);
                centerX.push_back((double)R.x + (double)R.width/2);
                centerY.push_back((double)R.y + (double)R.height/2);
                rectangle(imageFinal, R, Scalar(255,255,255), 2, 8, 0);
            }

        }
        if(selected.size()>1 && selected.size() < 5){
            Point cnt;
            Point cnt2;
            vector<double> newHeight;
            for ( int l = 0; l < selected.size(); l++)
            {newHeight.push_back((94/allHeight[l])*37);}
            if (newHeight[0] > newHeight[1])
            {maxDist = newHeight[0];}
            else
            {maxDist = newHeight[1];}
            double halfDist = (newHeight[0]+newHeight[1])/2;
            double centerBoth = (centerX[0] + centerX[1])/2;
            double centerFrame = (imageFinal.cols)/2;
            if(verbose){
            std::cout<<centerX[0]<<"\n";
            std::cout<<centerX[1]<<"\n";
            std::cout<<centerBoth<<"\n";
            std::cout<<imageFinal.rows<<"\n";
            std::cout<<imageFinal.cols<<"\n";
            }
            double pixelAway = centerBoth - centerFrame;
            double degreePerPixel = hFOV/imageFinal.rows;
            double newTheta = pixelAway*degreePerPixel;
            table->PutNumber("Distance", halfDist);
            table->PutNumber("Angle", newTheta);
            for (int j = 0; j < 2; j++)
            {
                cnt2.x = 25;
                cnt2.y = 25 + k;
                cnt.x = (imageFinal.cols/2);
                cnt.y = (imageFinal.rows/2)+k;
                putText(imageFinal,std::to_string(newHeight[j]), cnt, FONT_HERSHEY_SIMPLEX, 1, Scalar(255,255,255), 1, 8, false);
                putText(imageFinal,std::to_string(newTheta), cnt2, FONT_HERSHEY_SIMPLEX, 1, Scalar(255,255,255), 1, 8, false);
                k = k + 50;
            }
        }
        else {
            table->PutNumber("Distance", -1.0);
        }
        imshow("raw", frame);
        imshow("edges", imageFinal);
        if((char)waitKey(10) == 27) break;
    }
}







