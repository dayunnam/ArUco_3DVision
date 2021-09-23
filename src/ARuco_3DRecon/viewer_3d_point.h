#pragma once
#include <string>
#include <opencv2/opencv.hpp>

#define Rx(rx)      (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, cos(rx), -sin(rx), 0, sin(rx), cos(rx))
#define Ry(ry)      (cv::Mat_<double>(3, 3) << cos(ry), 0, sin(ry), 0, 1, 0, -sin(ry), 0, cos(ry))
#define Rz(rz)      (cv::Mat_<double>(3, 3) << cos(rz), -sin(rz), 0, sin(rz), cos(rz), 0, 0, 0, 1)

bool viewer_3d_point(const std::string xyz_file, const cv::Mat& camMatrix,const std::string type) {
    
    std::string file_name;
    if(type == "cam")
        file_name = xyz_file.substr(0, xyz_file.length() - 4) + "(cam)" + xyz_file.substr(xyz_file.length() - 4, 4);
    else if(type == "obj")
        file_name = xyz_file.substr(0, xyz_file.length() - 4) + "(obj)" + xyz_file.substr(xyz_file.length() - 4, 4);
    else {
        return false;
    }
    // Load a point cloud in the homogeneous coordinate
    FILE* f_in = fopen(file_name.c_str(), "rt");
    if (f_in == NULL) return -1;
    cv::Mat X, X_rgb;
    while (!feof(f_in))
    {
        if (type == "cam") {
            double x, y, z, n_x, n_y, n_z;
            if (fscanf(f_in, "%lf %lf %lf %lf %lf %lf", &x, &y, &z, &n_x, &n_y, &n_z) == 6) X.push_back(cv::Vec4d(x, y, z, 1));
            //std::cout << x << " " << " " << y << " " << z << std::endl;
        }
        else if (type == "obj") {
            double x, y, z;
            int R, G, B;
            if (fscanf(f_in, "%lf %lf %lf %d %d %d", &x, &y, &z, &R, &G, &B) == 6) {
                X.push_back(cv::Vec4d(x, y, z, 1));
                X_rgb.push_back(cv::Vec4i(R, G, B, 1));
            }
        }
    }
    fclose(f_in);
    X = X.reshape(1).t(); // Convert to a 4 x N matrix
    if (type == "obj")
        X_rgb = X_rgb.reshape(1).t(); // Convert to a 4 x N matrix
  
    fprintf(stdout, "# 3D point: %d\n", X.size());
    
    double f = camMatrix.at<double>(1, 1), cx = camMatrix.at<double>(0, 2), cy = camMatrix.at<double>(1, 2);
    cv::Mat K = (cv::Mat_<double>(3, 3) << f, 0, cx, 0, f, cy, 0, 0, 1);

    
    // The given camera configuration: focal length, principal point, image resolution, position, and orientation
    cv::Size img_res(static_cast<int>(camMatrix.at<double>(0, 2)*2), static_cast<int>(camMatrix.at<double>(1, 2) * 2));
    
    cv::Point3d cam_pos = cv::Point3d(0, 0, 0); // position
    cv::Point3d cam_ori = cv::Point3d(0, 0, 0);  //orientation
    // Generate images for viewer
    int waitTime = 5;
    while (true) {
        
        // Derive a projection matrix
        cv::Mat Rc = Rz(cam_ori.z) * Ry(cam_ori.y) * Rx(cam_ori.x);
        cv::Mat tc(cam_pos);
        cv::Mat Rt;
        cv::hconcat(Rc.t(), -Rc.t() * tc, Rt);
        cv::Mat P = K * Rt;

        // Project the points (c.f. OpenCV provides 'cv::projectPoints()' with consideration of distortion.)
        cv::Mat x = P * X;

        x.row(0) = x.row(0) / x.row(2);
        x.row(1) = x.row(1) / x.row(2);
        x.row(2) = 1;

        // Show and store the points
        cv::Mat image = cv::Mat::zeros(img_res, CV_8UC3);
        for (int c = 0; c < x.cols; c++)
        {
            
            cv::Point p(x.col(c).rowRange(0, 2));
            cv::Point3i rgb(255, 0, 255);
            if (type == "obj")
                rgb = cv::Point3i(X_rgb.col(c).rowRange(0, 3));
            if (p.x >= 0 && p.x < img_res.width && p.y >= 0 && p.y < img_res.height) {
                cv::circle(image, p, 2, cv::Scalar(rgb.z, rgb.y, rgb.x), -1);
            }

        }

        putText(image, "Press direction key to translate view. Press 'ESC' to finish",
            cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0));
        if (type == "cam")cv::imshow("Camase Position", image);
        else if(type == "obj") cv::imshow("3D Structure", image);
        char key = (char)cv::waitKey(waitTime);
        if (key == 27) break;

        int inKey = (int)cv::waitKeyEx(waitTime);
        switch (inKey) {
            fprintf(stdout, "cam_pose : %f %f %f\n", cam_pos.x, cam_pos.y, cam_pos.z);
        case 0x250000: // Press Left Key.... 
            cam_pos = cv::Point3d(cam_pos.x - 0.05, cam_pos.y - 0.01, cam_pos.z);
            cam_ori = cv::Point3d(cam_ori.x - CV_PI / 1800,cam_ori.y + CV_PI / 1800, cam_ori.z);
            break;
        case 0x270000: // Press Right Key.... 
             cam_pos = cv::Point3d(cam_pos.x + 0.05, cam_pos.y + 0.01, cam_pos.z);
             cam_ori = cv::Point3d(cam_ori.x + CV_PI / 1800, cam_ori.y - CV_PI / 1800, cam_ori.z);
            break;
           
        case 0x260000: // Press Up Key....
            cam_pos = cv::Point3d(cam_pos.x - 0.01, cam_pos.y + 0.05, cam_pos.z);
            cam_ori = cv::Point3d(cam_ori.x + CV_PI / 1800, cam_ori.y + CV_PI / 1800, cam_ori.z);
            break;
        case 0x280000: // Press Down Key....
            cam_pos = cv::Point3d(cam_pos.x + 0.01, cam_pos.y - 0.05, cam_pos.z);
            cam_ori = cv::Point3d(cam_ori.x - CV_PI / 1800, cam_ori.y - CV_PI / 1800, cam_ori.z);
            break;
        case 0x210000: // Press Page Up Key....
            cam_pos = cv::Point3d(cam_pos.x, cam_pos.y , cam_pos.z + 0.05);
            cam_ori = cv::Point3d(cam_ori.x, cam_ori.y, cam_ori.z);
            break;
        case 0x220000: // Press Page Down Key....
            cam_pos = cv::Point3d(cam_pos.x, cam_pos.y, cam_pos.z - 0.05);
            cam_ori = cv::Point3d(cam_ori.x , cam_ori.y, cam_ori.z);
            break;
        }

       
    }
}