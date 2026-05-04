#include "arcface.hpp"
#include <filesystem>
#include <numeric>

ArcFaceInference::ArcFaceInference(
    const std::string &model_path,
    std::shared_ptr<ModelDecryptor> decryptor,
    const std::string &device)
{
    try
    {
       	ov::Core core_; // OpenVINO core object

        std::string actual_model_path = model_path;
        std::string actual_weights_path = model_path.substr(0, model_path.size() - 3) + "bin";

        // Handle encrypted models if decryptor is provided
        if (decryptor) {
            // Check if the model is encrypted (has .enc extension)
            if (std::filesystem::exists(model_path + ".enc")) {
                // Decrypt both model and weights
                auto decrypted_paths = decryptor->decryptModelFiles(model_path);
                actual_model_path = decrypted_paths.model_path;
                actual_weights_path = decrypted_paths.weights_path;
                
                // Store paths for cleanup
                temp_files_.push_back(actual_model_path);
                temp_files_.push_back(actual_weights_path);
            }
        }

        // Validate model files exist

        if (!std::filesystem::exists(actual_model_path)) {
            throw std::runtime_error("ArcFace model file not found: " + actual_model_path);
        }

        // Load and configure model

        std::shared_ptr<ov::Model> model = core_.read_model(actual_model_path);


        // Configure dynamic batch size
        ov::PartialShape dynamic_shape = {
            ov::Dimension::dynamic(),
            INPUT_CHANNELS,
            INPUT_HEIGHT,
            INPUT_WIDTH};
        model->reshape({{INPUT_NAME, dynamic_shape}});

        // Compile model for specified device
        compiled_model_ = core_.compile_model(model, device);

        // Enable face processor debug output for development
        face_processor_.enableDebugOutput(false); // Set to true for debugging

        // std::cout << "ArcFace model initialized successfully on device: " << device << std::endl;
    }
    catch (const std::exception &e)
    {
        // Clean up any temporary files before throwing
        for (const auto& path : temp_files_) {
            std::filesystem::remove(path);
        }
        throw std::runtime_error("ArcFace initialization failed: " + std::string(e.what()));
    }
}

cv::Mat ArcFaceInference::run(const cv::Mat &input_image)
{
    try
    {
        // Enhanced input validation
        if (input_image.empty())
        {
            throw std::runtime_error("Input image is empty");
        }
        if (input_image.channels() != 3)
        {
            throw std::runtime_error("Input image must have 3 channels (BGR)");
        }
        if (input_image.type() != CV_8UC3)
        {
            throw std::runtime_error("Input image must be 8-bit unsigned integer (BGR)");
        }

        // Preprocess image
        cv::Mat preprocessed_img = face_processor_.preprocess_face(input_image, "ArcFace");

        // Create inference request and prepare input
        ov::InferRequest infer_request = compiled_model_.create_infer_request();
        auto input_port = compiled_model_.input();

        ov::Shape static_shape = {1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH};
        ov::Tensor input_tensor(input_port.get_element_type(), static_shape,
                                preprocessed_img.data);

        // Run inference
        infer_request.set_input_tensor(input_tensor);
        infer_request.infer();

        // Process output
        auto output = infer_request.get_output_tensor(0);
        const size_t output_size = shape_size(output.get_shape());
        float *output_buffer = output.data<float>();

        // Create embedding matrix
        return cv::Mat(1, output_size, CV_32F, output_buffer).clone();
    }
    catch (const std::exception &ex)
    {
        throw std::runtime_error("ArcFace inference failed: " + std::string(ex.what()));
    }
}

size_t ArcFaceInference::shape_size(const ov::Shape &shape) const
{
    return std::accumulate(shape.begin(), shape.end(), 1ull, std::multiplies<size_t>());
}