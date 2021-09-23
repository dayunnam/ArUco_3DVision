#include "load_images.h"
#include "sfm.hpp"
#include "viewer_3d_point.h"
#include <unordered_set>
#include "opencv2/opencv.hpp"
#include <string>

//usage example1 : --i=../dataset/pics -c=../dataset/cam_param.txt -o=../dataset/sfm_point.xyz
using namespace cv;
using namespace std;

namespace {
    const char* about = "3D Reconstruction";
    const char* keys =
        "{i        |       | Input folder}"
        "{o        |       | output file name}"
        "{c        |       | Camera intrinsic parameters. Needed for camera pose }";
}

static bool readCameraParameters(string filename, Mat& camMatrix, Mat& distCoeffs) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened())
        return false;
    fs["camera_matrix"] >> camMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    return true;
}

int main(int argc, char* argv[])
{
    //Parse arguments 
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);
    if (argc < 2) {
        parser.printMessage();
        fprintf(stdout, "Usage exmaple: ARuco_3DRecon.exe --i=[the path of input data] -c=[camera parameter file (txt, cfg)] -o=[output file (xyz)]\n");
        return -1;
    }
    String in_folder = parser.get<String>("i");
    if (!parser.has("i"))
        return -1;
    Mat camMatrix, distCoeffs;
    if (parser.has("c")) {
        bool readOk = readCameraParameters(parser.get<string>("c"), camMatrix, distCoeffs);
        if (!readOk) {
            cerr << "Invalid camera file" << endl;
            return 0;
        }
    }
    
    String out_file;
    if (parser.has("o"))
        out_file = parser.get<String>("o");
    else
        out_file = "sfm_3d_point.xyz";


    // Load images
    std::vector<cv::Mat> img_set;

    if (!Load_Images(in_folder, img_set)) {
        cerr << "Invalid input images" << endl;
        return -1;
    }
    //std::cout << "# of pictures : " << in_folder.size() << std::endl;

    //Estimation camera pose
    if (!pose_estimation(img_set, camMatrix, distCoeffs, out_file)) {
        cerr << "Failure to pose estimation" << endl;
    }

    //Viewer 3D point
    if (!viewer_3d_point(out_file, camMatrix, "cam")) {
        cerr << "Failure to 3D viewer" << endl;
    }
    if (!viewer_3d_point(out_file, camMatrix, "obj")) {
        cerr << "Failure to 3D viewer" << endl;
    }
    cout << "End of the program\n" << endl;
    return 0;
}
