#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>

class SuperPointONNX {
public:
    SuperPointONNX(const std::string &model_path,
                   int img_h, int img_w,
                   bool use_cuda = true);

    /**
     * Detect keypoints and compute descriptors.
     * @param gray        input grayscale image CV_8UC1
     * @param keypoints   output vector of keypoint locations
     * @param descriptors output CV_32F matrix (num_keypoints x descriptor_dim)
     */
    void detectAndCompute(const cv::Mat &gray,
                          std::vector<cv::Point2f> &keypoints,
                          cv::Mat &descriptors);

private:
    Ort::Env env_;
    Ort::Session session_;
    Ort::SessionOptions session_options_;

    int height_, width_;
    int stride_;          // typically 8 for SuperPoint
    int desc_dim_;        // typically 256
    float detect_threshold_;  // new member

    // Preprocess the gray CV_8UC1 image into a float tensor [1,1,H,W]
    std::vector<float> preprocess(const cv::Mat &gray);
};

