#include "facenet.hpp"
#include <filesystem>
#include <numeric>


FaceNetInference::FaceNetInference(
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
            // Check if the model is encrypted
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


        // // First, we validate that the model file exists
        // if (!std::filesystem::exists(model_path))
        // {
        //     throw std::runtime_error("FaceNet model file not found: " + model_path);
        // }


        // Validate model file existence
        if (!std::filesystem::exists(actual_model_path)) {
            throw std::runtime_error("FaceNet model file not found: " + actual_model_path);
        }



        // Initialize OpenVINO runtime and load the model
        std::shared_ptr<ov::Model> model = core_.read_model(actual_model_path);


        // Configure the model for dynamic batch processing
        // Note: FaceNet uses a different input shape than ArcFace (NHWC format)
        ov::PartialShape dynamic_shape = {
            ov::Dimension::dynamic(), // Dynamic batch size
            INPUT_HEIGHT,             // Height: 160
            INPUT_WIDTH,              // Width: 160
            INPUT_CHANNELS            // Channels: 3
        };

        // Reshape the model with the FaceNet-specific input name "input_1"
        model->reshape({{INPUT_NAME, dynamic_shape}});

        // Compile the model for the specified device
        compiled_model_ = core_.compile_model(model, device);

        // Configure the preprocessing pipeline
        face_processor_.enableDebugOutput(false); // Enable for debugging if needed

        // Log successful initialization
        // std::cout << "FaceNet model successfully initialized on device: " << device << std::endl;

        // Log model input and output information for verification
        for (const auto &input : compiled_model_.inputs())
        {
            // std::cout << "Input tensor name: " << input.get_any_name() << "\n";
            // std::cout << "Input shape: " << input.get_partial_shape() << "\n";
            // std::cout << "Input element type: " << input.get_element_type() << "\n";
        }

        for (const auto &output : compiled_model_.outputs())
        {
            // std::cout << "Output shape: " << output.get_partial_shape() << "\n";
            // std::cout << "Output element type: " << output.get_element_type() << "\n";
        }
    }
    catch (const std::exception &e)
    {
        // Clean up any temporary files before throwing
        for (const auto& path : temp_files_) {
            std::filesystem::remove(path);
        }
        throw std::runtime_error("FaceNet initialization failed: " + std::string(e.what()));
    }
}

cv::Mat FaceNetInference::run(const cv::Mat &input_image)
{
    try
    {
        // Comprehensive input validation
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

        // Process the image through our preprocessing pipeline
        // FaceNet uses different preprocessing parameters than ArcFace
        cv::Mat preprocessed_img = face_processor_.preprocess_face(
            input_image,
            "Facenet512" // This identifier ensures correct preprocessing for FaceNet
        );

        // Create an inference request
        ov::InferRequest infer_request = compiled_model_.create_infer_request();

        // Get the input port and prepare the input tensor
        auto input_port = compiled_model_.input();

        // Create the input tensor with FaceNet's specific shape requirements
        ov::Shape static_shape = {
            1,             // Batch size
            INPUT_HEIGHT,  // 160
            INPUT_WIDTH,   // 160
            INPUT_CHANNELS // 3
        };

        // Create and set the input tensor
        ov::Tensor input_tensor(
            input_port.get_element_type(),
            static_shape,
            preprocessed_img.data);
        infer_request.set_input_tensor(input_tensor);

        // Run inference
        infer_request.infer();

        // Retrieve and process the output
        auto output = infer_request.get_output_tensor(0);
        const size_t output_size = shape_size(output.get_shape());
        float *output_buffer = output.data<float>();

        // Create and return the embedding matrix
        // FaceNet produces 512-dimensional embeddings like ArcFace
        return cv::Mat(1, output_size, CV_32F, output_buffer).clone();
    }
    catch (const std::exception &ex)
    {
        throw std::runtime_error("FaceNet inference failed: " + std::string(ex.what()));
    }
}

size_t FaceNetInference::shape_size(const ov::Shape &shape) const
{
    // Calculate the total size of the tensor shape
    return std::accumulate(shape.begin(), shape.end(), 1ull, std::multiplies<size_t>());
}