#ifndef PREPROCESSING_HPP
#define PREPROCESSING_HPP

#include <opencv2/opencv.hpp>
#include <tuple>
#include <string>

class FaceProcessor {
public:
    // Add debug control
    void enableDebugOutput(bool enable) { debug_output_ = enable; }
    
    cv::Mat prepare_face(
        const cv::Mat& current_img,
        const std::tuple<int, int>& target_size,
        bool grayscale = false,
        bool to_rgb = false
    );
    
    cv::Mat to_torch_tensor(const cv::Mat& img);
    cv::Mat preprocess_face(const cv::Mat& img, const std::string& model_name);
    cv::Mat normalize_image(const cv::Mat& img);
    cv::Mat unsqueeze_batch_dim(const cv::Mat& img, const std::string& model_name);
    
private:
    bool debug_output_ = false;
    std::tuple<int, int> get_input_size(const std::string& model_name) const;
    void save_matrix(const cv::Mat& matrix, const std::string& filename);
};

#endif