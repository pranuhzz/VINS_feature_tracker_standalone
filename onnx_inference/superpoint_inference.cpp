// superpoint_inference.cpp
// A clean, robust implementation of the SuperPoint ONNX inference wrapper

#include "superpoint_inference.h"
#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <algorithm>
#include <iostream>

// ------------------------------------------------------------
// Constructor: initializes ONNX Runtime session
// ------------------------------------------------------------
SuperPointONNX::SuperPointONNX(
    const std::string& model_path,
    int img_h,
    int img_w,
    bool use_cuda)
  : env_(ORT_LOGGING_LEVEL_WARNING, "SuperPoint"),
    session_options_(),
    session_(nullptr),
    height_(img_h),
    width_(img_w),
    stride_(8),
    desc_dim_(256),
    detect_threshold_(0.015f)
{
    // Choose execution provider
    if (use_cuda) {
        std::cout << "[INFO] Using CUDA Execution Provider\n";
        OrtSessionOptionsAppendExecutionProvider_CUDA(session_options_, 0);
    } else {
        std::cout << "[INFO] Using CPU Execution Provider\n";
    }

    session_options_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    try {
        session_ = Ort::Session(env_, model_path.c_str(), session_options_);
    } catch (const Ort::Exception& e) {
        std::cerr << "[ERROR] Failed to load ONNX model: " << e.what() << std::endl;
        throw;
    }
}

// ------------------------------------------------------------
// Preprocess: convert grayscale CV_8UC1 to [0,1] float tensor
// ------------------------------------------------------------
std::vector<float> SuperPointONNX::preprocess(const cv::Mat& gray) {
    cv::Mat resized;
    if (gray.rows != height_ || gray.cols != width_) {
        cv::resize(gray, resized, cv::Size(width_, height_));
    } else {
        resized = gray;
    }

    cv::Mat float_img;
    resized.convertTo(float_img, CV_32F, 1.0f / 255.0f);

    std::vector<float> tensor;
    tensor.reserve(height_ * width_);
    for (int y = 0; y < height_; ++y) {
        const float* ptr = float_img.ptr<float>(y);
        tensor.insert(tensor.end(), ptr, ptr + width_);
    }
    return tensor;
}

// ------------------------------------------------------------
// detectAndCompute: run SuperPoint to extract keypoints + descriptors
// ------------------------------------------------------------
void SuperPointONNX::detectAndCompute(
    const cv::Mat& gray,
    std::vector<cv::Point2f>& keypoints,
    cv::Mat& descriptors)
{
    // 1) Preprocess
    auto input_data = preprocess(gray);
    std::array<int64_t, 4> input_shape{1, 1, height_, width_};
    auto mem_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        input_data.data(),
        input_data.size(),
        input_shape.data(),
        input_shape.size()
    );

    // 2) Inference
    const char* input_names[] = {"image"};
    const char* output_names[] = {"semi", "desc"};
    auto outputs = session_.Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 2
    );

    // 3) Extract raw outputs
    auto& semi_tensor = outputs[0];
    auto& desc_tensor = outputs[1];
    auto semi_shape = semi_tensor.GetTensorTypeAndShapeInfo().GetShape();

    int64_t C = semi_shape[1];
    int64_t Hc = semi_shape[2];
    int64_t Wc = semi_shape[3];
    int64_t Cminus = C - 1;

    float* semi_data = semi_tensor.GetTensorMutableData<float>();
    float* desc_data = desc_tensor.GetTensorMutableData<float>();

    // 4) Softmax over "semi" (excluding dustbin channel)
    std::vector<float> prob(Cminus * Hc * Wc);
    for (int h = 0; h < Hc; ++h) {
        for (int w = 0; w < Wc; ++w) {
            // find max for numerical stability
            float mx = -std::numeric_limits<float>::infinity();
            for (int c = 0; c < C; ++c) {
                mx = std::max(mx, semi_data[c * Hc * Wc + h * Wc + w]);
            }
            // compute exp
            float sum = 0;
            for (int c = 0; c < Cminus; ++c) {
                float e = std::exp(
                    semi_data[c * Hc * Wc + h * Wc + w] - mx);
                prob[c * Hc * Wc + h * Wc + w] = e;
                sum += e;
            }
            // normalize
            for (int c = 0; c < Cminus; ++c) {
                prob[c * Hc * Wc + h * Wc + w] /= sum;
            }
        }
    }

    // 5) Non-maximum suppression + thresholding
    keypoints.clear();
    for (int h = 0; h < Hc; ++h) {
        for (int w = 0; w < Wc; ++w) {
            for (int c = 0; c < Cminus; ++c) {
                float p = prob[c * Hc * Wc + h * Wc + w];
                if (p < detect_threshold_) continue;

                bool is_max = true;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        int nh = h + dy;
                        int nw = w + dx;
                        if (nh < 0 || nh >= Hc || nw < 0 || nw >= Wc) continue;
                        if (prob[c * Hc * Wc + nh * Wc + nw] > p) {
                            is_max = false;
                            break;
                        }
                    }
                    if (!is_max) break;
                }
                if (!is_max) continue;

                // decode keypoint position
                float y = h * stride_ + (c / stride_) + 0.5f;
                float x = w * stride_ + (c % stride_) + 0.5f;
                keypoints.emplace_back(x, y);
            }
        }
    }

    // 6) Descriptor interpolation (nearest-neighbor)
    int Hd = Hc;
    int Wd = Wc;
    int N = static_cast<int>(keypoints.size());
    descriptors.create(N, desc_dim_, CV_32F);

    for (int i = 0; i < N; ++i) {
        int px = static_cast<int>(keypoints[i].x / stride_);
        int py = static_cast<int>(keypoints[i].y / stride_);
        px = std::clamp(px, 0, Wd - 1);
        py = std::clamp(py, 0, Hd - 1);

        float* desc_row = descriptors.ptr<float>(i);
        for (int d = 0; d < desc_dim_; ++d) {
            desc_row[d] = desc_data[d * Hd * Wd + py * Wd + px];
        }
    }
}

