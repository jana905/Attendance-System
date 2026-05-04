#include <tuple>
#include <vector>
#include <stdexcept>
#include <fstream>
#include <filesystem>
#include "preprocessing.hpp"
#include <opencv2/opencv.hpp>

void FaceProcessor::save_matrix(const cv::Mat& matrix, const std::string& filename) {
    // Only save debug output if enabled
    if (!debug_output_) {
        return;
    }

    try {
        std::filesystem::create_directories("debug_outputs");
        std::ofstream file("debug_outputs/" + filename, std::ios::binary);
        
        if (!file) {
            throw std::runtime_error("Could not open file for writing: " + filename);
        }

        // Write matrix metadata
        int dims = matrix.dims;
        file.write(reinterpret_cast<const char*>(&dims), sizeof(dims));

        // Write dimension sizes
        for (int i = 0; i < dims; ++i) {
            int size = matrix.size[i];
            file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        }

        // Write matrix type and data
        int type = matrix.type();
        file.write(reinterpret_cast<const char*>(&type), sizeof(type));
        file.write(reinterpret_cast<const char*>(matrix.data), 
                  matrix.total() * matrix.elemSize());

    } catch (const std::exception& e) {
        // Don't throw from debug functions
        std::cerr << "Warning: Failed to save debug matrix: " << e.what() << std::endl;
    }
}

cv::Mat FaceProcessor::prepare_face(
    const cv::Mat &current_img,
    const std::tuple<int, int> &target_size,
    bool grayscale,
    bool to_rgb)
{
    if (current_img.empty())
    {
        throw std::invalid_argument("Input image is empty.");
    }

    cv::Mat processed_img = current_img.clone();

    // Convert to grayscale if required
    if (grayscale)
    {
        cv::cvtColor(processed_img, processed_img, cv::COLOR_BGR2GRAY);
        cv::cvtColor(processed_img, processed_img, cv::COLOR_GRAY2BGR);
    }

    int target_height = std::get<0>(target_size);
    int target_width = std::get<1>(target_size);

    // Resize while maintaining aspect ratio
    double factor = std::min(
        static_cast<double>(target_height) / processed_img.rows,
        static_cast<double>(target_width) / processed_img.cols);

    cv::Size dsize(
        static_cast<int>(processed_img.cols * factor),
        static_cast<int>(processed_img.rows * factor));

    cv::resize(processed_img, processed_img, dsize, 0, 0, cv::INTER_LINEAR);

    // Add symmetric padding
    int diff_h = target_height - processed_img.rows;
    int diff_w = target_width - processed_img.cols;

    cv::copyMakeBorder(
        processed_img, processed_img,
        diff_h / 2, diff_h - diff_h / 2,
        diff_w / 2, diff_w - diff_w / 2,
        cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    if (to_rgb)
    {
        cv::cvtColor(processed_img, processed_img, cv::COLOR_BGR2RGB);
    }
    // std::cout << "processed_img_shape" << processed_img.rows << processed_img.cols << processed_img.channels() <<"\n";
    save_matrix(processed_img, "prepared_face.bin");

    return processed_img;
}

cv::Mat FaceProcessor::to_torch_tensor(const cv::Mat &img)
{
    cv::Mat tensor_img;
    img.convertTo(tensor_img, CV_32F, 1 / 255.0); // Normalize to [0, 1]

    save_matrix(tensor_img, "normalized_image.bin");

    return tensor_img;
}

cv::Mat FaceProcessor::normalize_image(const cv::Mat &img)
{
    if (img.empty())
    {
        throw std::invalid_argument("Input image is empty.");
    }

    cv::Mat normalized_img;
    img.convertTo(normalized_img, CV_32F); // Ensure float data type

    std::vector<cv::Mat> channels(3);
    cv::split(normalized_img, channels);

    for (auto &channel : channels)
    {
        channel = (channel - 0.5f) / 0.5f; // Normalize to [-1, 1]
    }

    cv::merge(channels, normalized_img);

    save_matrix(normalized_img, "normalized_image.bin");

    return normalized_img;
}

cv::Mat FaceProcessor::unsqueeze_batch_dim(const cv::Mat &img, const std::string &model_name) {
    if (img.empty() || img.channels() != 3) {
        throw std::invalid_argument("Invalid input image.");
    }

    std::vector<cv::Mat> channels(3);
    cv::split(img, channels);
    cv::Mat chw_img;
    cv::vconcat(channels, chw_img);

    std::vector<cv::Mat> batch = {chw_img};
    cv::Mat batch_chw_img;
    cv::vconcat(batch, batch_chw_img);

    if (model_name == "ArcFace") {
        batch_chw_img = batch_chw_img.reshape(1, {1, 3, img.rows, img.cols});
    } else if (model_name == "Facenet512") {
        batch_chw_img = batch_chw_img.reshape(1, {1, img.rows, img.cols, 3});
    }

    save_matrix(batch_chw_img, "batch_dim.bin");
    return batch_chw_img;
}


cv::Mat FaceProcessor::preprocess_face(const cv::Mat &img, const std::string &model_name) {
    auto target_size = get_input_size(model_name);
    cv::Mat processed_img = prepare_face(img, target_size, false, false);
    cv::Mat tensor_img = to_torch_tensor(processed_img);
    
    if (model_name == "ArcFace") {
        cv::Mat normalized_img = normalize_image(tensor_img);
        return unsqueeze_batch_dim(normalized_img, model_name);
    } else if (model_name == "Facenet512") {
        return tensor_img;
    }
    throw std::runtime_error("Unsupported model type: " + model_name);
}


std::tuple<int, int> FaceProcessor::get_input_size(const std::string &model_name) const {
    if (model_name == "ArcFace") {
        return std::make_tuple(112, 112);
    } else if (model_name == "Facenet512") {
        return std::make_tuple(160, 160);
    }
    throw std::runtime_error("Unsupported model type: " + model_name);
}