#include <vector>
#include <opencv2/opencv.hpp>

bool Load_Images(const cv::String in_folder, std::vector<cv::Mat> & out_images) {
    std::vector<cv::String> img_names;
    try
    {
        cv::glob(in_folder, img_names, true);
        if (img_names.size() < 2)
            throw   img_names;
    }
    catch (cv::Exception)
    {
        std::cout << "There are no folder named " << in_folder << " in tihs directory" << std::endl;
        getchar();
        return false;
    }
    catch (std::vector<cv::String> img_names_)
    {
        std::cout << "Need more images in " << in_folder << std::endl;
        getchar();
        return false;
    }
    int num_images = static_cast<int>(img_names.size());

    if (num_images >= 1000) {
        std::cerr << "Too many pictures to process" << std::endl;
        return false;
    }
  
    out_images.resize(num_images);
    for (int i = 0; i < num_images; ++i)
    {
        out_images[i] = cv::imread(img_names[i]);

        if (out_images[i].empty())
        {
            std::cout << "Can't open image " << img_names[i] << std::endl;
            getchar();
            return -1;
        }
    }
    return true;
}