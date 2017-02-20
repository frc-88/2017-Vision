#include "networktables/NetworkTable.h"
#include "ntcore.h"
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <string>
#include <unistd.h>

using namespace cv;
using std::shared_ptr;
Mat frameB;
Mat frameG;
gpu::GpuMat gpuFrame;
int setHeight = 94;
int setDist = 37;
float maxDist;
float minDist;
float oldDist;
int distBetween = 6;
double hFOV = 62;


NetworkTable *table;

Mat visionProcessing (Mat gpuMAT, double hue, double sat, double val)
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
    Mat hsvThresholdInput(gpuhsvThresholdInput);
    inRange(hsvThresholdInput,Scalar(hue-10, sat-100, val-100), Scalar(hue+10, sat+100, val+100), hsvThresholdOutput);
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


Mat contouringGear(Mat contourInput, bool verbose, shared_ptr<NetworkTable> table)
{
    bool externalOnly = false;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    contours.clear();
    int mode = externalOnly ? RETR_EXTERNAL : RETR_LIST;
    int method = CHAIN_APPROX_SIMPLE;
    findContours(contourInput, contours, hierarchy, mode, method);
    float w_threshold = 250;
    float wl_threshold = 50;
    float h_threshold = 250;
    float hl_threshold = 5;
    vector<int> selected;
    vector<double> centerX;
    vector<double> centerY;
    vector<double> allHeight;
    vector<double> allWidth;
    Mat imageFinalG(contourInput);
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
            rectangle(imageFinalG, R, Scalar(255,255,255), 2, 8, 0);
        }

    }
    if(selected.size()>1 && selected.size() < 5){
        Point cnt;
        Point cnt2;
        vector<double> newHeight;
        for ( int l = 0; l < selected.size(); l++)
        {newHeight.push_back((94/allHeight[l])*37);}
        if (newHeight[0] > newHeight[1])
        {maxDist = newHeight[0]; minDist = newHeight[1];}
        else
        {maxDist = newHeight[1]; minDist = newHeight[0];}
        double newTheta = ((maxDist*maxDist)+(distBetween*distBetween)-(minDist*minDist))/(2*maxDist*distBetween);
        double halfDist = (newHeight[0]+newHeight[1])/2;
        double centerBoth = (centerX[0] + centerX[1])/2;
        double centerFrame = (imageFinalG.cols)/2;
        if(verbose){
            std::cout<<centerX[0]<<"\n";
            std::cout<<centerX[1]<<"\n";
            std::cout<<centerBoth<<"\n";
            std::cout<<imageFinalG.rows<<"\n";
            std::cout<<imageFinalG.cols<<"\n";
        }
        double pixelAway = centerBoth - centerFrame;
        double degreePerPixel = hFOV/imageFinalG.rows;
        double newGamma = pixelAway*degreePerPixel;
        table->PutNumber("DistanceG", halfDist);
        table->PutNumber("Gamma", newGamma);
        table->PutNumber("Theta", newTheta);
        for (int j = 0; j < 2; j++)
        {
            cnt2.x = 25;
            cnt2.y = 25 + k;
            cnt.x = (imageFinalG.cols/2);
            cnt.y = (imageFinalG.rows/2)+k;
            putText(imageFinalG,std::to_string(newHeight[j]), cnt, FONT_HERSHEY_SIMPLEX, 1, Scalar(255,255,255), 1, 8, false);
            putText(imageFinalG,std::to_string(newTheta), cnt2, FONT_HERSHEY_SIMPLEX, 1, Scalar(255,255,255), 1, 8, false);
            k = k + 50;
        }
    }
    else {
        table->PutNumber("DistanceG", -1.0);
    }
    return imageFinalG;
}


Mat contouringBoiler(Mat contourInput, bool verbose, shared_ptr<NetworkTable> table)
{
    bool externalOnly = false;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    contours.clear();
    int mode = externalOnly ? RETR_EXTERNAL : RETR_LIST;
    int method = CHAIN_APPROX_SIMPLE;
    findContours(contourInput, contours, hierarchy, mode, method);
    float w_threshold = 300;
    float wl_threshold = 30;
    float h_threshold = 300;
    float hl_threshold = 5;
    vector<int> selected;
    vector<double> centerX;
    vector<double> centerY;
    vector<double> allHeight;
    vector<double> allWidth;
    Mat imageFinalB(contourInput);
    int k = 0;
    for (int i = 0; i < contours.size(); i++)
    {
        if(verbose){
            std::cout<<contours.size()<<"\n";
        }
        Rect R = boundingRect(contours[i]);

        // filter contours according to their bounding box
        if (R.width < w_threshold && R.height < h_threshold && R.width > wl_threshold && R.height > hl_threshold)
            ///  if (contours.size()>0)
        {
            selected.push_back(i);
            allHeight.push_back((double)R.height);
            allWidth.push_back((double)R.width);
            centerX.push_back((double)R.x + (double)R.width/2);
            centerY.push_back((double)R.y + (double)R.height/2);
            rectangle(imageFinalB, R, Scalar(255,255,255), 2, 8, 0);
            //drawContours(imageFinalB, contours, i, Scalar(255,255,255), 2, 8, hierarchy, 0 ,Point() );
        }

    }
    if(selected.size()>1){
        Point cnt;
        Point cnt2;
        vector<double> newWidth;
        for ( int l = 0; l < selected.size(); l++)
        {newWidth.push_back((1.5*620)/(2*allWidth[l]*tan((3.14159/180)*hFOV/2)));}
        double halfDist = (newWidth[0]+newWidth[1])/2;
        double centerBoth = (centerX[0] + centerX[1])/2;
        double centerFrame = (imageFinalB.cols)/2;
        //        if(verbose){
        //            std::cout<<centerX[0]<<"\n";
        //            std::cout<<centerX[1]<<"\n";
        //            std::cout<<centerBoth<<"\n";
        //            std::cout<<imageFinalB.rows<<"\n";
        //            std::cout<<imageFinalB.cols<<"\n";
        //        }
        double pixelAway = centerBoth - centerFrame;
        double degreePerPixel = hFOV/imageFinalB.rows;
        double newTheta = pixelAway*degreePerPixel;
        table->PutNumber("DistanceB", halfDist);
        table->PutNumber("AngleB", newTheta);
        //std::cout<<halfDist<<"  "<<newTheta<<"\n";
        for (int j = 0; j < 2; j++)
        {
            cnt2.x = 25;
            cnt2.y = 25 + k;
            cnt.x = (imageFinalB.cols/2);
            cnt.y = (imageFinalB.rows/2)+k;
            putText(imageFinalB,std::to_string(halfDist), cnt, FONT_HERSHEY_SIMPLEX, 1, Scalar(255,255,255), 1, 8, false);
            putText(imageFinalB,std::to_string(newTheta), cnt2, FONT_HERSHEY_SIMPLEX, 1, Scalar(255,255,255), 1, 8, false);
            k = k + 50;
        }
    }
    else {
        table->PutNumber("DistanceB", -1.0);
        //std::cout<<"-1.0 \n";
    }
    return imageFinalB;
}


int main(int, char**)
{
    bool verbose = true;
    bool switchCam = true;
    int brightness = -1;
    int contrast = -1;
    int exposure = -1;
    int sharpness = -1;
    int saturation = -1;
    double temphue = -1;
    double tempsat = -1;
    double tempval = -1;
    double hue;
    double sat;
    double val;
    setDriver(brightness, contrast, exposure, sharpness, saturation);
    NetworkTable::SetClientMode();
    NetworkTable::SetTeam(88);
    NetworkTable::Initialize();
    shared_ptr<NetworkTable> table = NetworkTable::GetTable("imfeelinglucky");
    temphue = table->GetNumber("visionH");
    tempsat = table->GetNumber("visionS");
    tempval = table->GetNumber("visionV");
    camNum = table->GetNumber("visionFeed");
    switchCam = table->GetBoolean("camSwitch");
    if (temphue == -1)
    {
        hue = 50;
    }
    else
    {
        hue = temphue;
    }
    if (tempsat == -1)
    {
        sat = 150;
    }
    else
    {
        sat = tempsat;
    }
    if (tempval == -1)
    {
        val = 150;
    }
    else
    {
        val = tempval;
    }
    //table = NetworkTable::GetTable("imfeelinglucky");
    VideoCapture cap(-1); // open the default camera
    VideoCapture cap2(1);
    //VideoWriter outputVideo;
    if(!cap.isOpened())  // check if we succeeded
        return -1;
    if(!cap2.isOpened())  // check if we succeeded
        return -1;
    cap.set(CV_CAP_PROP_EXPOSURE, -200);
    cap.set(CV_CAP_PROP_FRAME_WIDTH,640);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT,480);
    cap2.set(CV_CAP_PROP_EXPOSURE, -200);
    cap2.set(CV_CAP_PROP_FRAME_WIDTH,640);
    cap2.set(CV_CAP_PROP_FRAME_HEIGHT,480);
    // namedWindow("raw", CV_WINDOW_AUTOSIZE);
    // namedWindow("edges",CV_WINDOW_AUTOSIZE);
    for(;;)
    {
        if (switchCam) {
            cap >> frameB; // get a new frame from boiler camera
            cap2 >> frameG; // get a new frame from gear camera
        }
        else {
            cap >> frameG; // get a new frame from gear camera
            cap2 >> frameB; // get a new frame from boiler camera

        }
        if (frameB.empty())
            break;
        Mat findContoursInputB;
        Mat findContoursInputG;
        Mat imageFinalB;
        Mat imageFinalG;
        findContoursInputB = visionProcessing(frameB, hue, sat, val);
        findContoursInputG = visionProcessing(frameG, hue, sat, val);
        imageFinalB = contouringBoiler(findContoursInputB, verbose, table);
        imageFinalG= contouringGear(findContoursInputG, verbose, table);
        switch(camNum) {
        case 1: imwrite("/home/ubuntu/stream/c1.jpg", imageFinalB); break;
        case 2: imwrite("/home/ubuntu/stream/c1.jpg", frameB); break;
        case 3: imwrite("/home/ubuntu/stream/c1.jpg", imageFinalG); break;
        case 4: imwrite("/home/ubuntu/stream/c1.jpg", frameG); break;
        }
        // imshow("raw", frameB);
        // imshow("edges", imageFinal);
        if((char)waitKey(10) == 27) break;
    }
}







