/*
* Reference: https://github.com/sunglok/3dv_tutorial
*/

#include "sfm.hpp"
#include "config.h"

#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/xfeatures2d.hpp"
#endif

class Img_info {
public:
    cv::Mat Image;
    std::string FileName;
    double focal; //focal
    double c_x, c_y; //c_x ,c_y
    double yaw, pitch, roll; //rotation
    double pos0, pos1, pos2; //position
    std::vector<cv::KeyPoint> features;
    int img_idx;
    double min_z_val; // minimium depth value
    double max_z_val; // maximum depth value
};

cv::Mat getCameraMat(const SFM::Vec9d& camera)
{
    const double& f = camera[6], & cx = camera[7], & cy = camera[8];
    cv::Mat K = (cv::Mat_<double>(3, 3) << f, 0, cx, 0, f, cy, 0, 0, 1);
    return K;
}

cv::Mat getProjectionMat(const SFM::Vec9d& camera)
{
    cv::Mat K = getCameraMat(camera);
    cv::Mat rvec = (cv::Mat_<double>(3, 1) << camera[0], camera[1], camera[2]), R, Rt;
    cv::Mat t = (cv::Mat_<double>(3, 1) << camera[3], camera[4], camera[5]);
    cv::Rodrigues(rvec, R);
    cv::hconcat(R, t, Rt);
    return K * Rt;
}

void updateCameraPose(SFM::Vec9d& camera, const cv::Mat& R, const cv::Mat& t)
{
    cv::Mat rvec = R.clone();
    if (rvec.cols == 3 && rvec.rows == 3) cv::Rodrigues(rvec, rvec);
    camera[0] = rvec.at<double>(0);
    camera[1] = rvec.at<double>(1);
    camera[2] = rvec.at<double>(2);
    camera[3] = t.at<double>(0);
    camera[4] = t.at<double>(1);
    camera[5] = t.at<double>(2);
}

bool isBadPoint(const cv::Point3d& X, const SFM::Vec9d& camera1, const SFM::Vec9d& camera2, double Z_limit, double max_cos_parallax)
{
    if (X.z < -Z_limit || X.z > Z_limit) return true;   // BAD! If the point is too far from the origin.
    cv::Vec3d rvec1(camera1[0], camera1[1], camera1[2]), rvec2(camera2[0], camera2[1], camera2[2]);
    cv::Matx33d R1, R2;
    cv::Rodrigues(rvec1, R1);
    cv::Rodrigues(rvec2, R2);
    cv::Point3d t1(camera1[3], camera1[4], camera1[5]), t2(camera2[3], camera2[4], camera2[5]);
    cv::Point3d p1 = R1 * X + t1;                       // A 3D point w.r.t. the 1st camera coordinate
    cv::Point3d p2 = R2 * X + t2;                       // A 3D point w.r.t. the 2nd camera coordinate
    if (p1.z <= 0 || p2.z <= 0) return true;            // BAD! If the point is beyond of one of 1st and 2nd cameras.
    cv::Point3d v2 = R1 * R2.t() * p2;                  // A 3D vector 'p2' w.r.t. the 1st camera coordinate
    double cos_parallax = p1.dot(v2) / cv::norm(p1) / cv::norm(v2);
    if (cos_parallax > max_cos_parallax) return true;   // BAD! If the point has small parallax angle.
    return false;
}

void RotationMatrixToEulerAngles(const cv::Matx33d& R_, double& yaw, double& pitch, double& roll) {
    cv::Matx33d permute(0, 0, 1.0, -1.0, 0, 0, 0, -1.0, 0);
    cv::Matx33d A = permute * R_;
    cv::Matx33d B = A * permute.t();
    cv::Matx33d R = B.t();

    yaw = 0.0;
    pitch = 0.0;
    roll = 0.0;

    if (std::fabs(R(0, 0)) < epsilon_ && std::fabs(R(1, 0)) < epsilon_) {
        yaw = std::atan2(R(1, 2), R(0, 2));
        if (std::fabs(R(2, 0)) < epsilon_) pitch = pi_ / 2.0;
        else pitch = -pi_ / 2.0;
        roll = 0.0;
    }
    else {
        yaw = std::atan2(R(1, 0), R(0, 0));
        if (std::fabs(R(0, 0)) < epsilon_) pitch = std::atan2(-R(2, 0), R(1, 0) / std::sin(yaw));
        else pitch = std::atan2(-R(2, 0), R(0, 0) / std::cos(yaw));

        roll = atan2(R(2, 1), R(2, 2));
    }

    yaw *= (180.0 / pi_);
    pitch *= (180.0 / pi_);
    roll *= (180.0 / pi_);
}

void TranslationToPosition(const cv::Vec3d& tvec, double scale, double& pos_0, double& pos_1, double& pos_2) {
    pos_0 = scale * tvec[2];
    pos_1 = -scale * tvec[0];
    pos_2 = -scale * tvec[1];
}

std::vector<bool> maskNoisyPoints(std::vector<cv::Point3d>& Xs, const std::vector<std::vector<cv::KeyPoint>>& xs, const std::vector<SFM::Vec9d>& views, const SFM::VisibilityGraph& visibility, double reproj_error2)
{
    std::vector<bool> is_noisy(Xs.size(), false);
    if (reproj_error2 > 0)
    {
        for (auto visible = visibility.begin(); visible != visibility.end(); visible++)
        {
            cv::Point3d& X = Xs[visible->second];
            if (X.z < 0) continue;
            int img_idx = SFM::getCamIdx(visible->first), pt_idx = SFM::getObsIdx(visible->first);
            const cv::Point2d& x = xs[img_idx][pt_idx].pt;
            const SFM::Vec9d& view = views[img_idx];

            // Project the given 'X'
            cv::Vec3d rvec(view[0], view[1], view[2]);
            cv::Matx33d R;
            cv::Rodrigues(rvec, R);
            cv::Point3d X_p = R * X + cv::Point3d(view[3], view[4], view[5]);
            const double& f = view[6], & cx = view[7], & cy = view[8];
            cv::Point2d x_p(f * X_p.x / X_p.z + cx, f * X_p.y / X_p.z + cy);

            // Calculate distance between 'x' and 'x_p'
            cv::Point2d d = x - x_p;
            if (d.x * d.x + d.y * d.y > reproj_error2) is_noisy[visible->second] = true;
        }
    }
    return is_noisy;
}

bool write_xyz_file(const cv::String out_file, const std::vector<cv::Point3d>& Xs, std::vector<cv::Vec3b> Xs_rgb, 
    const double Z_limit, std::vector<SFM::Vec9d> cameras, std::vector<bool> is_noisy = std::vector<bool>()) {
    // Store the 3D points to an XYZ file

    std::string cam_xyz_file_name = out_file.substr(0, out_file.length() - 4) + "(cam)" + out_file.substr(out_file.length() - 4, 4);
    std::string obj_xyz_file_name = out_file.substr(0, out_file.length() - 4) + "(obj)" + out_file.substr(out_file.length() - 4, 4);

    FILE* fpts = fopen(obj_xyz_file_name.c_str(), "wt");
    if (fpts == NULL) return -1;
    for (size_t i = 0; i < Xs.size(); i++) {
        if (Xs[i].z > -Z_limit && Xs[i].z < Z_limit) {
            if(!is_noisy.empty())
                if(is_noisy[i])
                fprintf(fpts, "%f %f %f %d %d %d\n", Xs[i].x, Xs[i].y, Xs[i].z, Xs_rgb[i][2], Xs_rgb[i][1], Xs_rgb[i][0]); // Format: x, y, z, R, G, B
            else
                fprintf(fpts, "%f %f %f %d %d %d\n", Xs[i].x, Xs[i].y, Xs[i].z, Xs_rgb[i][2], Xs_rgb[i][1], Xs_rgb[i][0]);
        }
    }
    fclose(fpts);

    // Store the camera poses to an XYZ file 
    FILE* fcam = fopen(cam_xyz_file_name.c_str(), "wt");
    if (fcam == NULL) return -1;
    for (size_t j = 0; j < cameras.size(); j++)
    {
        cv::Vec3d rvec(cameras[j][0], cameras[j][1], cameras[j][2]), t(cameras[j][3], cameras[j][4], cameras[j][5]);
        cv::Matx33d R;
        cv::Rodrigues(rvec, R);
        cv::Vec3d p = -R.t() * t;
        fprintf(fcam, "%f %f %f %f %f %f\n", p[0], p[1], p[2], R.t()(0, 2), R.t()(1, 2), R.t()(2, 2)); // Format: x, y, z, n_x, n_y, n_z
    }
    fclose(fcam);

}

bool sfm_global_without_camMatrix(const std::vector<cv::Mat>& img_set, const cv::String out_file) {

    int num_img = img_set.size();
    cv::Ptr<cv::Feature2D> finder;

    std::vector<cv::detail::ImageFeatures> feature_set(num_img);

    if (features_type == "orb")
    {
        finder = cv::ORB::create(2000);
    }
    else if (features_type == "akaze")
    {
        finder = cv::AKAZE::create();
    }
#ifdef HAVE_OPENCV_XFEATURES2D
    else if (features_type == "surf")
    {
        finder = cv::xfeatures2d::SURF::create(400);
    }
    else if (features_type == "sift") {
        finder = cv::SIFT::create();
    }
#endif
    else
    {
        std::cout << "Unknown 2D features type: '" << features_type << "'.\n";
        return -1;
    }
    for (int pic_idx = 0; pic_idx < num_img; pic_idx++) {
        cv::detail::computeImageFeatures(finder, img_set[pic_idx], feature_set[pic_idx]);
        feature_set[pic_idx].img_idx = pic_idx;
    }
    std::vector<cv::detail::MatchesInfo> pairwise_matches;
    cv::Ptr<cv::detail::FeaturesMatcher> matcher;

    if (matcher_type == "affine")
        matcher = cv::makePtr<cv::detail::AffineBestOf2NearestMatcher>(false, try_cuda, match_conf);
    else if (range_width == -1)
        matcher = cv::makePtr<cv::detail::BestOf2NearestMatcher>(try_cuda, match_conf);
    else
        matcher = cv::makePtr<cv::detail::BestOf2NearestRangeMatcher>(range_width, try_cuda, match_conf);

    (*matcher)(feature_set, pairwise_matches);
    matcher->collectGarbage();

    std::vector<int> indices = leaveBiggestComponent(feature_set, pairwise_matches, conf_thresh);
    int num_images = static_cast<int>(indices.size());

    if (num_images < 2)
    {
        std::cout << "Need more images" << std::endl;
        return false;
    }

    std::vector<Img_info> Image_info_(num_images);
    for (size_t i = 0; i < num_images; i++) {
        Image_info_[i].Image = img_set[indices[i]];
        Image_info_[i].img_idx = indices[i];
        Image_info_[i].features = feature_set[i].keypoints;
    }

    cv::Ptr<cv::detail::Estimator> estimator;
    if (estimator_type == "affine")
        estimator = cv::makePtr<cv::detail::AffineBasedEstimator>();
    else
        estimator = cv::makePtr<cv::detail::HomographyBasedEstimator>();

    std::vector<cv::detail::CameraParams> cameras_;
    if (!(*estimator)(feature_set, pairwise_matches, cameras_))
    {
        std::cout << "Homography estimation failed.\n";
        return false;
    }

    std::vector<double> focals;
    for (size_t i = 0; i < cameras_.size(); ++i)
    {
        //   std::cout << "Camera #" << indices[i] + 1 << ":\nK:\n" << cameras_[i].K() << "\nR:\n" << cameras_[i].R;
        focals.push_back(cameras_[i].focal);
    }

    sort(focals.begin(), focals.end());
    double focal_init, cx_init, cy_init;
    double Z_init = 2, Z_limit = 100, ba_loss_width = 9;

    if (focals.size() % 2 == 1)
        focal_init = focals[focals.size() / 2];
    else
        focal_init = focals[focals.size() / 2 - 1] + focals[focals.size() / 2] * 0.5f;

    cx_init = img_set.front().cols / 2;
    cy_init = img_set.front().rows / 2;


    std::vector<std::pair<uint, uint>> match_pair;
    std::vector<std::vector<cv::DMatch>> match_inlier;

    for (size_t i = 0; i < num_images; i++) {
        for (size_t j = i + 1; j < num_images; j++) {
            int pair_idx = i * num_images + j;

            if (pairwise_matches[pair_idx].confidence < conf_thresh) continue;

            std::vector<cv::DMatch> matches_ = pairwise_matches[pair_idx].matches;
            std::vector<cv::DMatch> inlier;
            std::vector<cv::Point2d> src, dst;

            for (auto itr = matches_.begin(); itr != matches_.end(); itr++)
            {
                src.push_back(Image_info_[i].features[itr->queryIdx].pt);
                dst.push_back(Image_info_[i].features[itr->trainIdx].pt);
            }
            for (size_t k = 0; k < pairwise_matches[pair_idx].matches.size() && pairwise_matches[pair_idx].matches.size() > 5; ++k)
            {
                if (!pairwise_matches[pair_idx].inliers_mask[k])
                    continue;
                inlier.push_back(matches_[k]);
            }

            fprintf(stdout, "Image %zd - %zd are matched (%zd / %zd).\n", i, j, inlier.size(), matches_.size());

            match_pair.push_back(std::make_pair(uint(i), uint(j)));
            match_inlier.push_back(inlier);

            if (show_match)
            {
                cv::Mat match_image;
                cv::drawMatches(Image_info_[i].Image, Image_info_[i].features, Image_info_[j].Image, Image_info_[j].features, inlier,
                    match_image, cv::Scalar::all(-1), cv::Scalar::all(-1));
                cv::imshow("Feature and Matches", match_image);
                cv::waitKey(2000);
            }
        }
    }

    if (match_pair.size() < 1) {
        std::cout << "the number of match_pair is less than 1\n";
        return false;
    }
    // 1) Initialize cameras (rotation, translation, intrinsic parameters)
    std::vector<SFM::Vec9d> cameras(img_set.size(), SFM::Vec9d(0, 0, 0, 0, 0, 0, focal_init, cx_init, cy_init));

    // 2) Initialize 3D points and build a visibility graph
    std::vector<cv::Point3d> Xs;
    std::vector<cv::Vec3b> Xs_rgb;
    SFM::VisibilityGraph xs_visited;
    for (size_t m = 0; m < match_pair.size(); m++)
    {
        for (size_t in = 0; in < match_inlier[m].size(); in++)
        {
            const uint& cam1_idx = match_pair[m].first, & cam2_idx = match_pair[m].second;
            const uint& x1_idx = match_inlier[m][in].queryIdx, & x2_idx = match_inlier[m][in].trainIdx;
            const uint key1 = SFM::genKey(cam1_idx, x1_idx), key2 = SFM::genKey(cam2_idx, x2_idx);
            auto visit1 = xs_visited.find(key1), visit2 = xs_visited.find(key2);
            if (visit1 != xs_visited.end() && visit2 != xs_visited.end())
            {
                // Remove previous observations if they are not consistent
                if (visit1->second != visit2->second)
                {
                    xs_visited.erase(visit1);
                    xs_visited.erase(visit2);
                }
                continue; // Skip if two observations are already visited
            }

            uint X_idx = 0;
            if (visit1 != xs_visited.end()) X_idx = visit1->second;
            else if (visit2 != xs_visited.end()) X_idx = visit2->second;
            else
            {
                // Add a new point if two observations are not visited
                X_idx = uint(Xs.size());
                Xs.push_back(cv::Point3d(0, 0, Z_init));
                Xs_rgb.push_back(img_set[cam1_idx].at<cv::Vec3b>(Image_info_[cam1_idx].features[x1_idx].pt));
            }
            if (visit1 == xs_visited.end()) xs_visited[key1] = X_idx;
            if (visit2 == xs_visited.end()) xs_visited[key2] = X_idx;
        }
    }
    printf("# of 3D points: %zd\n", Xs.size());

    // 3) Optimize camera pose and 3D points together (bundle adjustment)
    ceres::Problem ba;
    for (auto visit = xs_visited.begin(); visit != xs_visited.end(); visit++)
    {
        int cam_idx = SFM::getCamIdx(visit->first), x_idx = SFM::getObsIdx(visit->first);
        const cv::Point2d& x = Image_info_[cam_idx].features[x_idx].pt;
        SFM::addCostFunc6DOF(ba, Xs[visit->second], x, cameras[cam_idx], ba_loss_width);
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR; //SPARSE_SCHUR, ITERATIVE_SCHUR, SPARSE_NORMAL_CHOLESKY, SCHUR_JACOBI, DENSE_SCHUR
    options.max_num_iterations = 1000;
    options.preconditioner_type = ceres::JACOBI;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.logging_type = ceres::SILENT;
    options.parameter_tolerance = (1e-8);//
    options.num_threads = 6;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &ba, &summary);
    std::cout << summary.FullReport() << std::endl;

    if (summary.termination_type == ceres::FAILURE)
    {
        std::cout << "error: ceres failed.\n";
        exit(1);
    }
    std::vector<std::vector<cv::KeyPoint>> img_keypoint;
    img_keypoint.reserve(Image_info_.size());
    for (size_t mm = 0; mm < Image_info_.size(); mm++) {
        img_keypoint.emplace_back(Image_info_[mm].features);
    }

    // Mark erroneous points to reject them
    std::vector<bool> is_noisy = maskNoisyPoints(Xs, img_keypoint, cameras, xs_visited, ba_loss_width);
    int num_noisy = std::accumulate(is_noisy.begin(), is_noisy.end(), 0);

    //assert(Image_info_.size == cameras.size());

    write_xyz_file(out_file, Xs, Xs_rgb, Z_limit, cameras, is_noisy);

    for (size_t j = 0; j < cameras.size(); j++) {

        double min_z, max_z;
        cv::Vec3d rvec(cameras[j][0], cameras[j][1], cameras[j][2]), t(cameras[j][3], cameras[j][4], cameras[j][5]);
        cv::Matx33d R;
        cv::Rodrigues(rvec, R);
        cv::Vec3d p = -R.t() * t;
        cv::Vec3d cam2pt(Xs[0].x - p[0], Xs[0].y - p[1], Xs[0].z - p[2]);
        cv::Vec3d normal(R.t()(0, 2), R.t()(1, 2), R.t()(2, 2));

        // compute minimum and maximum of depth
        double z = cam2pt.dot(normal);
        min_z = z;
        max_z = z;
        for (size_t i = 1; i < Xs.size(); i++)
        {
            if (Xs[i].z > -Z_limit && Xs[i].z < Z_limit && !is_noisy[i]) {
                cv::Vec3d cam2pt(Xs[i].x - p[0], Xs[i].y - p[1], Xs[i].z - p[2]);
                cv::Vec3d normal(R.t()(0, 2), R.t()(1, 2), R.t()(2, 2));
                double z_temp = cam2pt.dot(normal);
                if (z_temp > 0) {
                    if (z_temp < min_z) { min_z = z_temp; }
                    else if (z_temp > max_z) { max_z = z_temp; }
                }
            }
        }

        Image_info_[j].max_z_val = max_z;
        Image_info_[j].min_z_val = min_z;

        RotationMatrixToEulerAngles(R, Image_info_[j].yaw, Image_info_[j].pitch, Image_info_[j].roll);
        TranslationToPosition(p, 1, Image_info_[j].pos0, Image_info_[j].pos1, Image_info_[j].pos2);

        Image_info_[j].focal = cameras[j][6];
        Image_info_[j].c_x = cameras[j][7];
        Image_info_[j].c_y = cameras[j][8];

        fprintf(stdout, "# of 3D points: %zd (Rejected: %d)\n", Xs.size(), num_noisy);
        fprintf(stdout, "Camera %zd's position (axis_0, axis_1, axis_2) = (%.3f, %.3f, %.3f)\n", j, Image_info_[j].pos0, Image_info_[j].pos1, Image_info_[j].pos2);
        fprintf(stdout, "Camera %zd's rotation (yaw, pitch, roll) = (%.3f, %.3f, %.3f)\n", j, Image_info_[j].yaw, Image_info_[j].pitch, Image_info_[j].roll);
        fprintf(stdout, "Camera %zd's (f, cx, cy) = (%.3f, %.3f, %.3f)\n", j, Image_info_[j].focal, Image_info_[j].c_x, Image_info_[j].c_y);
        fprintf(stdout, "Camera %zd's z range  = [%.3f (=min) , %.3f (=max)]\n\n", j, min_z, max_z);

        
    }

}

bool sfm_inc_without_camMatrix(const std::vector<cv::Mat>& img_set, const cv::String out_file) {

    int num_img = img_set.size();
    cv::Ptr<cv::Feature2D> finder;

    std::vector<cv::detail::ImageFeatures> feature_set(num_img);

    if (features_type == "orb")
    {
        finder = cv::ORB::create(2000);
    }
    else if (features_type == "akaze")
    {
        finder = cv::AKAZE::create();
    }
#ifdef HAVE_OPENCV_XFEATURES2D
    else if (features_type == "surf")
    {
        finder = cv::xfeatures2d::SURF::create(400);
    }
    else if (features_type == "sift") {
        finder = cv::SIFT::create();
    }
#endif
    else
    {
        std::cout << "Unknown 2D features type: '" << features_type << "'.\n";
        return -1;
    }
    for (int pic_idx = 0; pic_idx < num_img; pic_idx++) {
        cv::detail::computeImageFeatures(finder, img_set[pic_idx], feature_set[pic_idx]);
        feature_set[pic_idx].img_idx = pic_idx;
    }
    std::vector<cv::detail::MatchesInfo> pairwise_matches;
    cv::Ptr<cv::detail::FeaturesMatcher> matcher;

    if (matcher_type == "affine")
        matcher = cv::makePtr<cv::detail::AffineBestOf2NearestMatcher>(false, try_cuda, match_conf);
    else if (range_width == -1)
        matcher = cv::makePtr<cv::detail::BestOf2NearestMatcher>(try_cuda, match_conf);
    else
        matcher = cv::makePtr<cv::detail::BestOf2NearestRangeMatcher>(range_width, try_cuda, match_conf);

    (*matcher)(feature_set, pairwise_matches);
    matcher->collectGarbage();

    std::vector<int> indices = leaveBiggestComponent(feature_set, pairwise_matches, conf_thresh);
    int num_images = static_cast<int>(indices.size());

    if (num_images < 2)
    {
        std::cout << "Need more images" << std::endl;
        return false;
    }

    std::vector<Img_info> Image_info_(num_images);
    for (size_t i = 0; i < num_images; i++) {
        Image_info_[i].Image = img_set[indices[i]];
        Image_info_[i].img_idx = indices[i];
        Image_info_[i].features = feature_set[i].keypoints;
    }

    cv::Ptr<cv::detail::Estimator> estimator;
    if (estimator_type == "affine")
        estimator = cv::makePtr<cv::detail::AffineBasedEstimator>();
    else
        estimator = cv::makePtr<cv::detail::HomographyBasedEstimator>();

    std::vector<cv::detail::CameraParams> cameras_;
    if (!(*estimator)(feature_set, pairwise_matches, cameras_))
    {
        std::cout << "Homography estimation failed.\n";
        return false;
    }

    std::vector<double> focals;
    for (size_t i = 0; i < cameras_.size(); ++i)
    {
        //   std::cout << "Camera #" << indices[i] + 1 << ":\nK:\n" << cameras_[i].K() << "\nR:\n" << cameras_[i].R;
        focals.push_back(cameras_[i].focal);
    }

    sort(focals.begin(), focals.end());
    double focal_init, cx_init, cy_init;
    double Z_init = 2, Z_limit = 100, ba_loss_width = 9;

    if (focals.size() % 2 == 1)
        focal_init = focals[focals.size() / 2];
    else
        focal_init = focals[focals.size() / 2 - 1] + focals[focals.size() / 2] * 0.5f;

    cx_init = img_set.front().cols / 2;
    cy_init = img_set.front().rows / 2;

    std::vector<std::pair<uint, uint>> match_pair;
    std::vector<std::vector<cv::DMatch>> match_inlier;

    for (size_t i = 0; i < num_images; i++) {
        for (size_t j = i + 1; j < num_images; j++) {
            int pair_idx = i * num_images + j;

            if (pairwise_matches[pair_idx].confidence < conf_thresh) continue;

            std::vector<cv::DMatch> matches_ = pairwise_matches[pair_idx].matches;
            std::vector<cv::DMatch> inlier;
            std::vector<cv::Point2d> src, dst;

            for (auto itr = matches_.begin(); itr != matches_.end(); itr++)
            {
                src.push_back(Image_info_[i].features[itr->queryIdx].pt);
                dst.push_back(Image_info_[i].features[itr->trainIdx].pt);
            }
            for (size_t k = 0; k < pairwise_matches[pair_idx].matches.size() && pairwise_matches[pair_idx].matches.size() > 5; ++k)
            {
                if (!pairwise_matches[pair_idx].inliers_mask[k])
                    continue;
                inlier.push_back(matches_[k]);
            }

            fprintf(stdout, "Image %zd - %zd are matched (%zd / %zd).\n", i, j, inlier.size(), matches_.size());

            match_pair.push_back(std::make_pair(uint(i), uint(j)));
            match_inlier.push_back(inlier);

            if (show_match)
            {
                cv::Mat match_image;
                cv::drawMatches(Image_info_[i].Image, Image_info_[i].features, Image_info_[j].Image, Image_info_[j].features, inlier,
                    match_image, cv::Scalar::all(-1), cv::Scalar::all(-1));
                cv::imshow("Feature and Matches", match_image);
                cv::waitKey(2000);
            }
        }
    }

    if (match_pair.size() < 1) {
        std::cout << "the number of match_pair is less than 1\n";
        return false;
    }

    std::vector<SFM::Vec9d> cameras(img_set.size(), SFM::Vec9d(0, 0, 0, 0, 0, 0, focal_init, cx_init, cy_init));

    //-----------------------------------------------------------------------------------
    //¿€º∫ ¡ﬂ 
}

bool sfm_inc_with_camMatrix(const std::vector<cv::Mat>& img_set, const cv::Mat camMatrix, const cv::Mat distCoeffs, const cv::String out_file) {

    cv::Ptr<cv::FeatureDetector> fdetector = cv::BRISK::create();
    std::vector<std::vector<cv::KeyPoint>> img_keypoint;
    std::vector<cv::Mat> img_descriptor;

    for (int i = 0; i < img_set.size(); ++i)
    {
        std::vector<cv::KeyPoint> keypoint;
        cv::Mat descriptor;
        fdetector->detectAndCompute(img_set[i], cv::Mat(), keypoint, descriptor);
        img_keypoint.push_back(keypoint);
        img_descriptor.push_back(descriptor.clone());
    }

    if (img_set.size() < 2) return -1;

    double Z_limit = 100, max_cos_parallax = cos(10 * CV_PI / 180), ba_loss_width = 9; // Negative 'loss_width' makes BA not to use a loss function.
    int min_inlier_num = 200, ba_num_iter = 200; // Negative 'ba_num_iter' uses the default value for BA minimization

    // Match features and find good matches
    cv::Ptr<cv::DescriptorMatcher> fmatcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    std::vector<std::pair<uint, uint>> match_pair;        // Good matches (image pairs)
    std::vector<std::vector<cv::DMatch>> match_inlier;  // Good matches (inlier feature matches)
    for (size_t i = 0; i < img_set.size(); i++)
    {
        for (size_t j = i + 1; j < img_set.size(); j++)
        {
            // Match features of two image pair (i, j) and find their inliers
            std::vector<cv::DMatch> match, inlier;
            fmatcher->match(img_descriptor[i], img_descriptor[j], match);
            std::vector<cv::Point2d> src, dst;
            for (auto itr = match.begin(); itr != match.end(); itr++)
            {
                src.push_back(img_keypoint[i][itr->queryIdx].pt);
                dst.push_back(img_keypoint[j][itr->trainIdx].pt);
            }
            cv::Mat inlier_mask;
            cv::findFundamentalMat(src, dst, inlier_mask, cv::RANSAC);
            for (int k = 0; k < inlier_mask.rows; k++)
                if (inlier_mask.at<uchar>(k)) inlier.push_back(match[k]);
            //fprintf(stdout, "image %zd - %zd are matched (%zd / %zd).\n", i, j, inlier.size(), match.size());

            // Determine whether the image pair is good or not
            if (inlier.size() < min_inlier_num) continue;
            //fprintf(stdout, "Image %zd - %zd are selected.\n", i, j);
            match_pair.push_back(std::make_pair(uint(i), uint(j)));
            match_inlier.push_back(inlier);

            if (show_match)
            {
                cv::Mat match_image;
                cv::drawMatches(img_set[i], img_keypoint[i], img_set[j], img_keypoint[j], match, match_image, cv::Scalar::all(-1), cv::Scalar::all(-1), inlier_mask);
                cv::imshow("Structure-from-Motion", match_image);
                cv::waitKey();
            }
        }
    }
    if (match_pair.size() < 4) return -1;

    // Initialize cameras (rotation, translation, intrinsic parameters)
    std::vector<SFM::Vec9d> cameras(img_set.size(), SFM::Vec9d(0, 0, 0, 0, 0, 0, camMatrix.at<double>(1, 1), camMatrix.at<double>(0, 2), camMatrix.at<double>(1, 2)));

    uint best_pair = 0;
    std::vector<uint> best_score(match_inlier.size());
    for (size_t i = 0; i < match_inlier.size(); i++)
        best_score[i] = uint(match_inlier[i].size());
    cv::Mat best_Xs;
    while (true)
    {
        // 1) Select the best pair
        for (size_t i = 0; i < best_score.size(); i++)
            if (best_score[i] > best_score[best_pair]) best_pair = uint(i);
        if (best_score[best_pair] == 0)
        {
            printf("There is no good match. Try again after reducing 'max_cos_parallax'.\n");
            return -1;
        }
        const uint best_cam0 = match_pair[best_pair].first, best_cam1 = match_pair[best_pair].second;;

        // 2) Estimate relative pose from the best two views (epipolar geometry)
        std::vector<cv::Point2d> src, dst;
        for (auto itr = match_inlier[best_pair].begin(); itr != match_inlier[best_pair].end(); itr++)
        {
            src.push_back(img_keypoint[best_cam0][itr->queryIdx].pt);
            dst.push_back(img_keypoint[best_cam1][itr->trainIdx].pt);
        }
        cv::Mat K = getCameraMat(cameras[best_cam0]), R, t, inlier_mask;
        cv::Mat E = cv::findEssentialMat(src, dst, K, cv::RANSAC, 0.999, 1, inlier_mask);
        cv::recoverPose(E, src, dst, K, R, t, inlier_mask);
        for (int r = inlier_mask.rows - 1; r >= 0; r--)
        {
            if (!inlier_mask.at<uchar>(r))
            {
                // Remove additionally detected outliers
                src.erase(src.begin() + r);
                dst.erase(dst.begin() + r);
                match_inlier[best_pair].erase(match_inlier[best_pair].begin() + r);
            }
        }
        updateCameraPose(cameras[best_cam1], R, t);

        // 3) Reconstruct 3D points of the best two views (triangulation)
        cv::Mat P0 = getProjectionMat(cameras[best_cam0]), P1 = getProjectionMat(cameras[best_cam1]);
        cv::triangulatePoints(P0, P1, src, dst, best_Xs);
        best_Xs.row(0) = best_Xs.row(0) / best_Xs.row(3);
        best_Xs.row(1) = best_Xs.row(1) / best_Xs.row(3);
        best_Xs.row(2) = best_Xs.row(2) / best_Xs.row(3);

        best_score[best_pair] = 0;
        for (int i = 0; i < best_Xs.cols; i++)
        {
            cv::Point3d p(best_Xs.col(i).rowRange(0, 3)); // A 3D point at 'idx'
            if (isBadPoint(p, cameras[best_cam0], cameras[best_cam1], Z_limit, max_cos_parallax)) continue; // Do not add if it is bad
            best_score[best_pair]++;
        }
        printf("Image %u - %u were checked as the best match (# of inliers = %zd, # of good points = %d).\n", best_cam0, best_cam1, match_inlier[best_pair].size(), best_score[best_pair]);
        if (best_score[best_pair] > 100) break;
        best_score[best_pair] = 0;
    } // End of the 1st 'while (true)'
    const uint best_cam0 = match_pair[best_pair].first, best_cam1 = match_pair[best_pair].second;;

    // Prepare the initial 3D points
    std::vector<cv::Point3d> Xs;
    Xs.reserve(10000); // Allocate memory in advance not to break pointer access in Ceres Solver
    std::vector<cv::Vec3b> Xs_rgb;
    SFM::VisibilityGraph xs_visited;
    for (int i = 0; i < best_Xs.cols; i++)
    {
        cv::Point3d p(best_Xs.col(i).rowRange(0, 3)); // A 3D point at 'idx'
        if (isBadPoint(p, cameras[best_cam0], cameras[best_cam1], Z_limit, max_cos_parallax)) continue; // Do not add if it is bad
        uint X_idx = uint(Xs.size()), x0_idx = match_inlier[best_pair][i].queryIdx, x1_idx = match_inlier[best_pair][i].trainIdx;
        Xs.push_back(p);
        Xs_rgb.push_back(img_set[best_cam0].at<cv::Vec3b>(img_keypoint[best_cam0][x0_idx].pt));
        xs_visited[SFM::genKey(best_cam0, x0_idx)] = X_idx;
        xs_visited[SFM::genKey(best_cam1, x1_idx)] = X_idx;
    }
    std::unordered_set<uint> img_added;
    img_added.insert(best_cam0);
    img_added.insert(best_cam1);
    printf("Image %d - %d are complete (# of 3D points = %zd).\n", best_cam0, best_cam1, Xs.size());

    // Prepare bundle adjustment
    ceres::Problem ba;
    for (auto visit = xs_visited.begin(); visit != xs_visited.end(); visit++)
    {
        int cam_idx = SFM::getCamIdx(visit->first), x_idx = SFM::getObsIdx(visit->first);
        const cv::Point2d& x = img_keypoint[cam_idx][x_idx].pt;
        SFM::addCostFunc6DOF(ba, Xs[visit->second], x, cameras[cam_idx], ba_loss_width);
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    if (ba_num_iter > 0) options.max_num_iterations = ba_num_iter;
    options.num_threads = 8;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;

    while (true)
    {
        // 4) Select the next image to add
        std::vector<uint> img_score(img_set.size(), 0);
        std::vector<std::vector<uint>> match_table(img_set.size());
        for (size_t img = 0; img < img_score.size(); img++)
        {
            if (img_added.find(uint(img)) == img_added.end())                                                           // When the image is not added to the viewing graph
            {
                for (size_t i = 0; i < match_pair.size(); i++)
                {
                    if (match_pair[i].first == img && img_added.find(match_pair[i].second) != img_added.end())          // When 'first' is the current image and 'second' is already added
                    {
                        for (auto itr = match_inlier[i].begin(); itr != match_inlier[i].end(); itr++)
                            if (xs_visited.find(SFM::genKey(match_pair[i].second, itr->trainIdx)) != xs_visited.end())  // When a matched inlier is in 'Xs', the current image gains more score
                                img_score[img]++;
                        match_table[img].push_back(uint(i));
                    }
                    else if (match_pair[i].second == img && img_added.find(match_pair[i].first) != img_added.end())     // When 'second' is the current image and 'first' is already added
                    {
                        for (auto itr = match_inlier[i].begin(); itr != match_inlier[i].end(); itr++)
                            if (xs_visited.find(SFM::genKey(match_pair[i].first, itr->queryIdx)) != xs_visited.end())   // When a matched inlier is in 'Xs', the current image gains more score
                                img_score[img]++;
                        match_table[img].push_back(uint(i));
                    }
                }
            }
        }
        const auto next_score = std::max_element(img_score.begin(), img_score.end());
        const uint next_cam = static_cast<uint>(std::distance(img_score.begin(), next_score));
        const std::vector<uint> next_match = match_table[next_cam];
        if (next_match.empty()) break;

        // Separate points into known (pts_*) and unknown (new_*) for PnP (known) and triangulation (unknown)
        std::vector<cv::Point3d> pts_3d;
        std::vector<cv::Point2d> pts_2d;
        std::vector<uint> pts_key, pts_idx, new_pair_cam;
        std::vector<std::vector<cv::Point2d>> new_next_2d(next_match.size()), new_pair_2d(next_match.size());
        std::vector<std::vector<uint>> new_next_key(next_match.size()), new_pair_key(next_match.size());
        for (size_t i = 0; i < next_match.size(); i++)
        {
            bool next_is_first = true;
            int pair_cam = match_pair[next_match[i]].second;
            if (pair_cam == next_cam)
            {
                next_is_first = false;
                pair_cam = match_pair[next_match[i]].first;
            }
            new_pair_cam.push_back(pair_cam);
            for (auto itr = match_inlier[next_match[i]].begin(); itr != match_inlier[next_match[i]].end(); itr++)
            {
                int next_idx = itr->queryIdx, pair_idx = itr->trainIdx;
                if (!next_is_first)
                {
                    next_idx = itr->trainIdx;
                    pair_idx = itr->queryIdx;
                }
                auto found = xs_visited.find(SFM::genKey(pair_cam, pair_idx));
                if (found != xs_visited.end())  // When the matched point is already known (--> PnP)
                {
                    pts_3d.push_back(Xs[found->second]);
                    pts_2d.push_back(img_keypoint[next_cam][next_idx].pt);
                    pts_key.push_back(SFM::genKey(next_cam, next_idx));
                    pts_idx.push_back(found->second);
                }
                else                            // When the matched point is newly observed (--> triangulation)
                {
                    new_next_2d[i].push_back(img_keypoint[next_cam][next_idx].pt);
                    new_pair_2d[i].push_back(img_keypoint[pair_cam][pair_idx].pt);
                    new_next_key[i].push_back(SFM::genKey(next_cam, next_idx));
                    new_pair_key[i].push_back(SFM::genKey(pair_cam, pair_idx));
                }
            }
        }

        // 5) Estimate relative pose of the next view (PnP)
        if (pts_3d.size() < 10)
        {
            printf("Image %d is ignored (due to the small number of points).\n", next_cam);
            img_added.insert(next_cam);
            continue;
        }
        cv::Mat K = getCameraMat(cameras[next_cam]), rvec, t;
        std::vector<int> inlier_idx;
        cv::solvePnPRansac(pts_3d, pts_2d, K, cv::Mat::zeros(5, 1, CV_64F), rvec, t, false, 100, 4.0, 0.999, inlier_idx);
        updateCameraPose(cameras[next_cam], rvec, t);
        for (size_t i = 0; i < pts_key.size(); i++)
        {
            SFM::addCostFunc6DOF(ba, Xs[pts_idx[i]], pts_2d[i], cameras[next_cam], ba_loss_width);
            xs_visited[pts_key[i]] = pts_idx[i];
        }

        // 6) Reconstruct newly observed 3D points (triangulation)
        uint new_pts_total = 0;
        for (auto new_pts = new_next_2d.begin(); new_pts != new_next_2d.end(); new_pts++)
            new_pts_total += uint(new_pts->size());
        if (new_pts_total < 10)
        {
            printf("Image %d is complete (only localization; no 3D point addition).\n", next_cam);
            img_added.insert(next_cam);
            continue;
        }
        cv::Mat P0 = getProjectionMat(cameras[next_cam]);
        for (size_t i = 0; i < new_pair_cam.size(); i++)
        {
            const int pair_cam = new_pair_cam[i];
            cv::Mat P1 = getProjectionMat(cameras[pair_cam]), new_Xs;
            cv::triangulatePoints(P0, P1, new_next_2d[i], new_pair_2d[i], new_Xs);
            new_Xs.row(0) = new_Xs.row(0) / new_Xs.row(3);
            new_Xs.row(1) = new_Xs.row(1) / new_Xs.row(3);
            new_Xs.row(2) = new_Xs.row(2) / new_Xs.row(3);

            for (int j = 0; j < new_Xs.cols; j++)
            {
                cv::Point3d p(new_Xs.col(j).rowRange(0, 3));
                if (isBadPoint(p, cameras[next_cam], cameras[pair_cam], Z_limit, max_cos_parallax)) continue; // Do not add if it is bad
                uint X_idx = uint(Xs.size());
                Xs.push_back(p);
                Xs_rgb.push_back(img_set[next_cam].at<cv::Vec3b>(new_next_2d[i][j]));
                xs_visited[new_next_key[i][j]] = X_idx;
                xs_visited[new_pair_key[i][j]] = X_idx;
            }
        }
        printf("Image %d is complete (# of 3D points = %zd).\n", next_cam, Xs.size());

        // 7) Optimize camera pose and 3D points together (bundle adjustment)
        ceres::Solve(options, &ba, &summary);
        img_added.insert(next_cam);
    } // End of the 2nd 'while (true)'
    for (size_t j = 0; j < cameras.size(); j++)
        printf("Camera %zd's (f, cx, cy) = (%.3f, %.1f, %.1f)\n", j, cameras[j][6], cameras[j][7], cameras[j][8]);

    write_xyz_file(out_file, Xs, Xs_rgb, Z_limit, cameras);
    

}

bool sfm_global_with_camMatrix(const std::vector<cv::Mat>& img_set, const cv::Mat camMatrix, const cv::Mat distCoeffs, const cv::String out_file) {
    
    double Z_init = 2, Z_limit = 100, ba_loss_width = 9; // Negative 'loss_width' makes BA not to use a loss function.
    int min_inlier_num = 200, ba_num_iter = 200; // Negative 'ba_num_iter' uses the default value for BA minimization

    // Extract features
    cv::Ptr<cv::FeatureDetector> fdetector = cv::BRISK::create();
    std::vector<std::vector<cv::KeyPoint>> img_keypoint;
    std::vector<cv::Mat> img_descriptor;
    //std::vector<cv::Mat> images(img_set.size());
    //cv::Mat map1, map2;
    for(int i = 0 ; i < img_set.size(); i++)
    {
        /*
        if (!distCoeffs.empty()) {
            if (map1.empty() || map2.empty())
                cv::initUndistortRectifyMap(camMatrix, distCoeffs, cv::Mat(), cv::Mat(), img_set[i].size(), CV_32FC1, map1, map2);
            cv::remap( img_set[i], images[i], map1, map2, cv::InterpolationFlags::INTER_LINEAR);
            
            cv::imshow("org", img_set[i]);
            cv::imshow("cal", images[i]);
            cv::waitKey(3000);
        }*/

        std::vector<cv::KeyPoint> keypoint;
        cv::Mat descriptor;
        fdetector->detectAndCompute(img_set[i], cv::Mat(), keypoint, descriptor);
        img_keypoint.push_back(keypoint);
        img_descriptor.push_back(descriptor.clone());
    }
    if (img_set.size() < 2) return -1;

    // Match features and find good matches
    cv::Ptr<cv::DescriptorMatcher> fmatcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    std::vector<std::pair<uint, uint>> match_pair;        // Good matches (image pairs)
    std::vector<std::vector<cv::DMatch>> match_inlier;  // Good matches (inlier feature matches)
    for (size_t i = 0; i < img_set.size(); i++)
    {
        for (size_t j = i + 1; j < img_set.size(); j++)
        {
            // Match features of two image pair (i, j) and find their inliers
            std::vector<cv::DMatch> match, inlier;
            fmatcher->match(img_descriptor[i], img_descriptor[j], match);
            std::vector<cv::Point2d> src, dst;
            for (auto itr = match.begin(); itr != match.end(); itr++)
            {
                src.push_back(img_keypoint[i][itr->queryIdx].pt);
                dst.push_back(img_keypoint[j][itr->trainIdx].pt);
            }
            cv::Mat inlier_mask;
            cv::findFundamentalMat(src, dst, inlier_mask, cv::RANSAC);
            for (int k = 0; k < inlier_mask.rows; k++)
                if (inlier_mask.at<uchar>(k)) inlier.push_back(match[k]);
            printf("Image %zd - %zd are matched (%zd / %zd).\n", i, j, inlier.size(), match.size());

            // Determine whether the image pair is good or not
            if (inlier.size() < min_inlier_num) continue;
            printf("Image %zd - %zd are selected.\n", i, j);
            match_pair.push_back(std::make_pair(uint(i), uint(j)));
            match_inlier.push_back(inlier);
            if (show_match)
            {
                cv::Mat match_image;
                cv::drawMatches(img_set[i], img_keypoint[i], img_set[j], img_keypoint[j], match, match_image, cv::Scalar::all(-1), cv::Scalar::all(-1), inlier_mask);
                cv::imshow("Structure-from-Motion", match_image);
                cv::waitKey();
            }
        }
    }
    if (match_pair.size() < 1) return -1;

    // 1) Initialize cameras (rotation, translation, intrinsic parameters)
    std::vector<SFM::Vec9d> cameras(img_set.size(), SFM::Vec9d(0, 0, 0, 0, 0, 0, camMatrix.at<double>(1, 1), camMatrix.at<double>(0, 2), camMatrix.at<double>(1, 2)));

    // 2) Initialize 3D points and build a visibility graph
    std::vector<cv::Point3d> Xs;
    std::vector<cv::Vec3b> Xs_rgb;
    SFM::VisibilityGraph xs_visited;
    for (size_t m = 0; m < match_pair.size(); m++)
    {
        for (size_t in = 0; in < match_inlier[m].size(); in++)
        {
            const uint& cam1_idx = match_pair[m].first, & cam2_idx = match_pair[m].second;
            const uint& x1_idx = match_inlier[m][in].queryIdx, & x2_idx = match_inlier[m][in].trainIdx;
            const uint key1 = SFM::genKey(cam1_idx, x1_idx), key2 = SFM::genKey(cam2_idx, x2_idx);
            auto visit1 = xs_visited.find(key1), visit2 = xs_visited.find(key2);
            if (visit1 != xs_visited.end() && visit2 != xs_visited.end())
            {
                // Remove previous observations if they are not consistent
                if (visit1->second != visit2->second)
                {
                    xs_visited.erase(visit1);
                    xs_visited.erase(visit2);
                }
                continue; // Skip if two observations are already visited
            }

            uint X_idx = 0;
            if (visit1 != xs_visited.end()) X_idx = visit1->second;
            else if (visit2 != xs_visited.end()) X_idx = visit2->second;
            else
            {
                // Add a new point if two observations are not visited
                X_idx = uint(Xs.size());
                Xs.push_back(cv::Point3d(0, 0, Z_init));
                Xs_rgb.push_back(img_set[cam1_idx].at<cv::Vec3b>(img_keypoint[cam1_idx][x1_idx].pt));
            }
            if (visit1 == xs_visited.end()) xs_visited[key1] = X_idx;
            if (visit2 == xs_visited.end()) xs_visited[key2] = X_idx;
        }
    }
    printf("# of 3D points: %zd\n", Xs.size());

    // 3) Optimize camera pose and 3D points together (bundle adjustment)
    ceres::Problem ba;
    for (auto visit = xs_visited.begin(); visit != xs_visited.end(); visit++)
    {
        int cam_idx = SFM::getCamIdx(visit->first), x_idx = SFM::getObsIdx(visit->first);
        const cv::Point2d& x = img_keypoint[cam_idx][x_idx].pt;
        SFM::addCostFunc6DOF(ba, Xs[visit->second], x, cameras[cam_idx], ba_loss_width);
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    if (ba_num_iter > 0) options.max_num_iterations = ba_num_iter;
    options.num_threads = 8;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &ba, &summary);
    std::cout << summary.FullReport() << std::endl;

    // Mark erroneous points to reject them
    std::vector<bool> is_noisy = maskNoisyPoints(Xs, img_keypoint, cameras, xs_visited, ba_loss_width);
    int num_noisy = std::accumulate(is_noisy.begin(), is_noisy.end(), 0);
    printf("# of 3D points: %zd (Rejected: %d)\n", Xs.size(), num_noisy);
    for (size_t j = 0; j < cameras.size(); j++)
        printf("Camera %zd's (f, cx, cy) = (%.3f, %.1f, %.1f)\n", j, cameras[j][6], cameras[j][7], cameras[j][8]);

    write_xyz_file(out_file, Xs, Xs_rgb, Z_limit,  cameras, is_noisy);
    
}

bool pose_estimation(const std::vector<cv::Mat>& img_set, const cv::Mat camMatrix, const cv::Mat distCoeffs, const cv::String out_file) {
   
    if (camMatrix.empty()) {
        if(sfm_type == "sfm_global")
            return sfm_global_without_camMatrix(img_set, out_file);
        //if (sfm_type == "sfm_inc")
        //    return sfm_inc_without_camMatrix(img_set, out_file);
    }
    else {
        if (sfm_type == "sfm_global")
            return sfm_global_with_camMatrix(img_set, camMatrix, distCoeffs,  out_file);
        if (sfm_type == "sfm_inc")
            return sfm_inc_with_camMatrix(img_set, camMatrix, distCoeffs, out_file);
    }

    return false;
}