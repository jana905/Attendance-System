#pragma once
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include "preprocessing.hpp"
#include "model_decryptor.hpp"

#include <filesystem>

class FaceNetInference {
public:
    explicit FaceNetInference(
        const std::string& model_path,

        std::shared_ptr<ModelDecryptor> decryptor = nullptr,

        const std::string& device = "AUTO"
    );
    
    cv::Mat run(const cv::Mat& input_image);
    

   ~FaceNetInference() {
        // Cleanup temporary files
        for (const auto& path : temp_files_) {
            // std::filesystem::remove(path);
          std::filesystem::remove(std::filesystem::path(path));

        }
        compiled_model_ = ov::CompiledModel();
        core_ = ov::Core();
    }
    
    FaceNetInference(const FaceNetInference&) = delete;
    FaceNetInference& operator=(const FaceNetInference&) = delete;



private:
    ov::Core core_;
    ov::CompiledModel compiled_model_;
    FaceProcessor face_processor_;
    
    std::shared_ptr<ModelDecryptor> decryptor_;
    std::vector<std::string> temp_files_;
    
    const unsigned int INPUT_HEIGHT = 160;
    const unsigned int INPUT_WIDTH = 160;
    const unsigned int INPUT_CHANNELS = 3;
    const std::string INPUT_NAME = "input_1";
    
    size_t shape_size(const ov::Shape& shape) const;
};
