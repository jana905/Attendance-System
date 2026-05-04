#pragma once
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include "preprocessing.hpp"
#include "model_decryptor.hpp"
#include <filesystem>

class ArcFaceInference
{
public:
    // Constructor now accepts device selection
    explicit ArcFaceInference(
        const std::string &model_path,
        std::shared_ptr<ModelDecryptor> decryptor = nullptr,
        const std::string &device = "AUTO");

    // Main inference method with enhanced input validation
    cv::Mat run(const cv::Mat &input_image);

    // Proper cleanup of OpenVINO resources

    ~ArcFaceInference() {
        // Cleanup temporary files if they exist
        for (const auto& path : temp_files_) {
            // std::filesystem::remove(path);
            std::filesystem::remove(std::filesystem::path(path));

        }
        compiled_model_ = ov::CompiledModel();
        core_ = ov::Core();
    }

    ArcFaceInference(const ArcFaceInference &) = delete;
    ArcFaceInference &operator=(const ArcFaceInference &) = delete;


private:
    ov::Core core_;
    ov::CompiledModel compiled_model_;
    FaceProcessor face_processor_;
    std::shared_ptr<ModelDecryptor> decryptor_;
    std::vector<std::string> temp_files_; // Track temporary files for cleanup


    // Model-specific parameters
    unsigned long INPUT_CHANNELS = 3; // Example value, replace as needed
    unsigned long INPUT_HEIGHT = 112; // Example value, replace as needed
    unsigned long INPUT_WIDTH = 112;  // Example value, replace as needed
    const std::string INPUT_NAME = "x";

    size_t shape_size(const ov::Shape &shape) const;
};