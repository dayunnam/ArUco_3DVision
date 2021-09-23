
#include <string>

double epsilon_ = 0.0000001;
double pi_ = 3.1415926535;
std::string features_type = "orb";
float match_conf = 0.5f;
std::string matcher_type = "homography";
std::string estimator_type = "homography";
bool try_cuda = false;
int range_width = -1;
float conf_thresh = 1.f;
bool show_match = false;
std::string sfm_type = "sfm_global"; // "sfm_inc" or "sfm_global" 