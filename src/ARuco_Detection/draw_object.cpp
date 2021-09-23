#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;


class cube
{
private:
	vector< cv::Point3f > cube_obj_pts;
	vector< cv::Point2f > cube_img_pts;

public:
	cube();
	cv::Mat drawcube(cv::Mat, cv::Mat, cv::Mat, cv::Vec3d, cv::Vec3d);
};

cube::cube()
{
	Point3f c1(-0.05, -0.05, 0.05),
		c2(0.05, -0.05, 0.05),
		c3(0.05, 0.05, 0.05),
		c4(-0.05, 0.05, 0.05),
		c5(-0.05, -0.05, -0.05),
		c6(0.05, -0.05, -0.05),
		c7(0.05, 0.05, -0.05),
		c8(-0.05, 0.05, -0.05);
	cube_obj_pts.push_back(c1);
	cube_obj_pts.push_back(c2);
	cube_obj_pts.push_back(c3);
	cube_obj_pts.push_back(c4);
	cube_obj_pts.push_back(c5);
	cube_obj_pts.push_back(c6);
	cube_obj_pts.push_back(c7);
	cube_obj_pts.push_back(c8);
}

cv::Mat cube::drawcube(cv::Mat frame, cv::Mat intrinsic_matrix, cv::Mat distortion_parameters, cv::Vec3d rvecs, cv::Vec3d tvecs)
{

	cv::projectPoints(cube_obj_pts, rvecs, tvecs, intrinsic_matrix, distortion_parameters, cube_img_pts);

	for (int i = 0; i < cube_img_pts.size(); i++)
		circle(frame, cube_img_pts[i], 2, Scalar(0, 255, 0), 2, 8);
	line(frame, cube_img_pts[0], cube_img_pts[1], Scalar(0, 0, 255), 2, LINE_AA);
	line(frame, cube_img_pts[1], cube_img_pts[2], Scalar(0, 0, 255), 2, LINE_AA);
	line(frame, cube_img_pts[2], cube_img_pts[3], Scalar(0, 0, 255), 2, LINE_AA);
	line(frame, cube_img_pts[3], cube_img_pts[0], Scalar(0, 0, 255), 2, LINE_AA);
	line(frame, cube_img_pts[0], cube_img_pts[4], Scalar(0, 0, 255), 2, LINE_AA);
	line(frame, cube_img_pts[1], cube_img_pts[5], Scalar(0, 0, 255), 2, LINE_AA);
	line(frame, cube_img_pts[2], cube_img_pts[6], Scalar(0, 0, 255), 2, LINE_AA);
	line(frame, cube_img_pts[3], cube_img_pts[7], Scalar(0, 0, 255), 2, LINE_AA);
	line(frame, cube_img_pts[4], cube_img_pts[5], Scalar(0, 0, 255), 2, LINE_AA);
	line(frame, cube_img_pts[5], cube_img_pts[6], Scalar(0, 0, 255), 2, LINE_AA);
	line(frame, cube_img_pts[6], cube_img_pts[7], Scalar(0, 0, 255), 2, LINE_AA);
	line(frame, cube_img_pts[7], cube_img_pts[4], Scalar(0, 0, 255), 2, LINE_AA);

	return frame;
}