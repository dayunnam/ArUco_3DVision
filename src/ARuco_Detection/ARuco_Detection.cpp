

#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/aruco.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include "draw_object.h"

using namespace std;
using namespace cv;

//usage example0 --c=../dataset/cam_param.txt --d=10 -ci=1 --ov=../dataset/AR_detection.avi --o=../dataset/pics    //webcam
//usage example1 --c=../dataset/cam_param.txt --d=10 -ci=0 --ov=../dataset/AR_detection.avi --o=../dataset/pics   //laptap camera

namespace {
    const char* about = "Basic marker detection";
    const char* keys =
        "{d        |10     | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
        "DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
        "DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
        "DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16}"
        "{iv        |       | Input from video file, if ommited, input comes from camera }"
        "{ov        |       | output video file}"
        "{o        |       | folder to save captured pictures}"
        "{ci       | 0     | Camera id if input doesnt come from video (-v) }"
        "{c        |       | Camera intrinsic parameters. Needed for camera pose }"
        "{l        | 0.1   | Marker side lenght (in meters). Needed for correct scale in camera pose }";
}

static bool readCameraParameters(string filename, Mat& camMatrix, Mat& distCoeffs) {
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened())
        return false;
    fs["camera_matrix"] >> camMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    return true;
}


int main(int argc, char* argv[]) {


    CommandLineParser parser(argc, argv, keys);
    parser.about(about);
    cube cube1;
    if (argc < 2) {
        parser.printMessage();
        fprintf(stdout, "usage example : --c=../dataset/cam_param.txt --d=10 -ci=1 --ov=../dataset/AR_result.avi\n");
        return 0;
    }

    int dictionaryId = parser.get<int>("d");
    bool estimatePose = parser.has("c");
    float markerLength = parser.get<float>("l");

    Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
    detectorParams->cornerRefinementMethod = aruco::CORNER_REFINE_SUBPIX; // do corner refinement in markers

    int camId = parser.get<int>("ci");

    String in_video, out_video;
    if (parser.has("iv")) {
        in_video = parser.get<String>("iv");
    }
    if (!parser.check()) {
        parser.printErrors();
        return 0;
    }

    Ptr<aruco::Dictionary> dictionary =
        aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

    Mat camMatrix, distCoeffs;
    if (estimatePose) {
        bool readOk = readCameraParameters(parser.get<string>("c"), camMatrix, distCoeffs);
        if (!readOk) {
            cerr << "Invalid camera file" << endl;
            return 0;
        }
    }

    VideoCapture cap;


    int waitTime;
    if (!in_video.empty()) {
        cap.open(in_video);
        waitTime = 0;
    }
    else {
        cap.open(camId);
        waitTime = 100;
    }
    string outputFile = "./result.avi";
    string outputFolder = "..";
    VideoWriter video_writer;
    if (parser.has("ov")) {
        outputFile = parser.get<String>("ov");
        video_writer.open(outputFile, VideoWriter::fourcc('M', 'J', 'P', 'G'), 28, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
     }
    if (parser.has("o")) {
        outputFolder = parser.get<String>("o");
    }


    double totalTime = 0;
    int totalIterations = 0;
    bool draw_axis = true;
    int selected_image_num = 0;

    int selected_id = 0;
    bool select_marker = false;

    while (cap.grab()) {
        Mat image, imageCopy;
        
        cap.retrieve(image);
        image.copyTo(imageCopy);
        
        double tick = (double)getTickCount();

        vector< int > ids;
        vector< vector< Point2f > > corners, rejected;
        vector< Vec3d > rvecs, tvecs;


        // detect markers and estimate pose
        aruco::detectMarkers(image, dictionary, corners, ids, detectorParams, rejected);


        if (estimatePose && ids.size() > 0)
            aruco::estimatePoseSingleMarkers(corners, markerLength, camMatrix, distCoeffs, rvecs,
                tvecs);

        double currentTime = ((double)getTickCount() - tick) / getTickFrequency();
        totalTime += currentTime;
        totalIterations++;
        if (totalIterations % 30 == 0) {
            cout << "Detection Time = " << currentTime * 1000 << " ms "
                << "(Mean = " << 1000 * totalTime / double(totalIterations) << " ms)" << endl;
        }

        // draw results
        
        if (ids.size() > 0) {
            aruco::drawDetectedMarkers(imageCopy, corners, ids);

            if (estimatePose) {
                for (unsigned int i = 0; i < ids.size(); i++) {
                    if (draw_axis) {
                        aruco::drawAxis(imageCopy, camMatrix, distCoeffs, rvecs[i], tvecs[i],
                            markerLength * 0.5f);
                        continue;
                    }

                    if (ids[i] == selected_id) {
                        imageCopy = cube1.drawcube(imageCopy, camMatrix, distCoeffs, rvecs[i], tvecs[i]);
                    }
                }
            }
            if (!draw_axis) {
                putText(imageCopy, "Press 'Tab' key to show 3D Axis",
                    cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0));
            }
            else {
                putText(imageCopy, "Press 'Tab' key to show 3D Cube",
                    cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0));
            }

        }
        putText(imageCopy, "Press 'ESC' key to finish the program",
            cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0));
       
       
        char key = (char)waitKey(waitTime);
        if (key == 27) break;
        
        if (key == 9) {
            if (!draw_axis) {
                cout << "axis generated" << endl;
                putText(imageCopy, "Press 'Tab' key to show 3D Cube",
                    cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0));
            }
            else {
                cout << "axis  removed" << endl;
                putText(imageCopy, "Press 'Tab' key to show 3D Axis",
                    cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0));
                if (!select_marker) {
                    selected_id = ids[0];
                    select_marker = true;
                }
            }
            draw_axis = 1 ^ draw_axis;
        }
        if (key == 'c') {
            cout << "Frame captured" << endl;
            putText(imageCopy, "Frame captured",
                cv::Point(20, 80), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255),2);
            cv::String out_view = outputFolder + "/" + cv::format("view % 02d.png", selected_image_num);
            imwrite(out_view, image);
            selected_image_num++;
            
        }
        video_writer.write(imageCopy);
        imshow("ARuco_Detection", imageCopy);
        if (key == 'c')
            cv::waitKey(500);
    }
    cout << "Output file is stored as " << outputFile << endl;
    cap.release();
    video_writer.release();

    return 0;
}

