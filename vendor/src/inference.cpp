#include "inference.hpp"

#include <memory>
#include <opencv2/dnn.hpp>
#include <random>

namespace yolo {




Inference::Inference(
    const std::string &model_path,
	const cv::Size model_input_shape,
    const float &model_confidence_threshold,
    const float &model_NMS_threshold,
    std::shared_ptr<ModelDecryptor> decryptor)
    : model_input_shape_(cv::Size(640, 640)),
      model_confidence_threshold_(model_confidence_threshold),
      model_NMS_threshold_(model_NMS_threshold),
      decryptor_(decryptor)
{
    InitializeModel(model_path);
}




void Inference::InitializeModel(const std::string &model_path) {
    try {
        std::string actual_model_path = model_path;
        std::string actual_weights_path = model_path.substr(0, model_path.size() - 3) + "bin";
		
		ov::Core core_; // OpenVINO core object

        
        std::cout << "Initializing model with paths:" << std::endl;
        std::cout << "Model XML: " << actual_model_path << std::endl;
        std::cout << "Weights bin: " << actual_weights_path << std::endl;



        // Handle encrypted models if decryptor is provided
        if (decryptor_ && std::filesystem::exists(model_path + ".enc")) {
            std::cout << "Encrypted files found, decrypting..." << std::endl;

		    auto decrypted_paths = decryptor_->decryptModelFiles(model_path);
            actual_model_path = decrypted_paths.model_path;
            actual_weights_path = decrypted_paths.weights_path;
                        
            std::cout << "Decrypted paths:" << std::endl;
            std::cout << "Model XML: " << actual_model_path << std::endl;
            std::cout << "Weights bin: " << actual_weights_path << std::endl;
            

            // Store paths for cleanup
            temp_files_.push_back(actual_model_path);
            temp_files_.push_back(actual_weights_path);
        }

   // Validate model files exist and are readable
        if (!std::filesystem::exists(actual_model_path)) {
            throw std::runtime_error("Model file not found: " + actual_model_path);
        }
        if (!std::filesystem::exists(actual_weights_path)) {
            throw std::runtime_error("Weights file not found: " + actual_weights_path);
        }

        // Check file sizes
        std::cout << "Model file size: " << std::filesystem::file_size(actual_model_path) << " bytes" << std::endl;
        std::cout << "Weights file size: " << std::filesystem::file_size(actual_weights_path) << " bytes" << std::endl;


        // Read and configure the model
        std::shared_ptr<ov::Model> model = core_.read_model(actual_model_path);

        // Configure input shape
        if (model->is_dynamic()) {
            model->reshape({1, 3, 
                static_cast<long int>(model_input_shape_.height),
                static_cast<long int>(model_input_shape_.width)});
        }

        // Setup preprocessing
        ov::preprocess::PrePostProcessor ppp(model);
        ppp.input()
           .tensor()
           .set_element_type(ov::element::u8)
           .set_layout("NHWC")
           .set_color_format(ov::preprocess::ColorFormat::BGR);
        
        ppp.input()
           .preprocess()
           .convert_element_type(ov::element::f32)
           .convert_color(ov::preprocess::ColorFormat::RGB)
           .scale({255, 255, 255});
        
        ppp.input().model().set_layout("NCHW");
        ppp.output().tensor().set_element_type(ov::element::f32);
        
        model = ppp.build();

        // Compile the model
        compiled_model_ = core_.compile_model(model, "AUTO");
        inference_request_ = compiled_model_.create_infer_request();

        // Update shapes
        const auto& input_shape = model->inputs()[0].get_shape();
        model_input_shape_ = cv::Size2f(input_shape[2], input_shape[1]);

        const auto& output_shape = model->outputs()[0].get_shape();
        model_output_shape_ = cv::Size(output_shape[2], output_shape[1]);
    }
    catch (const std::exception &e) {
        // Clean up temporary files on failure
        for (const auto& path : temp_files_) {
            std::filesystem::remove(path);
        }
        throw std::runtime_error("YOLO model initialization failed: " + std::string(e.what()));
    }
}



// Method to run inference on an input frame
void Inference::RunInference(cv::Mat &frame) {
	Preprocessing(frame); // Preprocess the input frame
	inference_request_.infer(); // Run inference
	cropped_face = PostProcessing(frame); // Postprocess the inference results
}

// Method to preprocess the input frame
void Inference::Preprocessing(const cv::Mat &frame) {
	cv::Mat resized_frame;
	cv::resize(frame, resized_frame, model_input_shape_, 0, 0, cv::INTER_AREA); // Resize the frame to match the model input shape

	// Calculate scaling factor
	scale_factor_.x = static_cast<float>(frame.cols / model_input_shape_.width);
	scale_factor_.y = static_cast<float>(frame.rows / model_input_shape_.height);

	float *input_data = (float *)resized_frame.data; // Get pointer to resized frame data
	const ov::Tensor input_tensor = ov::Tensor(compiled_model_.input().get_element_type(), compiled_model_.input().get_shape(), input_data); // Create input tensor
	inference_request_.set_input_tensor(input_tensor); // Set input tensor for inference
}

// Method to postprocess the inference results
cv::Mat Inference::PostProcessing(cv::Mat &frame) {
	std::vector<int> class_list;
	std::vector<float> confidence_list;
	std::vector<cv::Rect> box_list;

	// Get the output tensor from the inference request
	const float *detections = inference_request_.get_output_tensor().data<const float>();
	const cv::Mat detection_outputs(model_output_shape_, CV_32F, (float *)detections); // Create OpenCV matrix from output tensor
	cv::Mat cropped_face;
	// Iterate over detections and collect class IDs, confidence scores, and bounding boxes
	for (int i = 0; i < detection_outputs.cols; ++i) {
		const cv::Mat classes_scores = detection_outputs.col(i).row(4);

		cv::Point class_id;
		double score;
		cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id); // Find the class with the highest score

		// Check if the detection meets the confidence threshold
		if (score > model_confidence_threshold_) {
			class_list.push_back(class_id.y);
			confidence_list.push_back(score);

			const float x = detection_outputs.at<float>(0, i);
			const float y = detection_outputs.at<float>(1, i);
			const float w = detection_outputs.at<float>(2, i);
			const float h = detection_outputs.at<float>(3, i);

			cv::Rect box;
			box.x = static_cast<int>(x);
			box.y = static_cast<int>(y);
			box.width = static_cast<int>(w);
			box.height = static_cast<int>(h);
			box_list.push_back(box);
		}
	}

	// Apply Non-Maximum Suppression (NMS) to filter overlapping bounding boxes
	std::vector<int> NMS_result;
	cv::dnn::NMSBoxes(box_list, confidence_list, model_confidence_threshold_, model_NMS_threshold_, NMS_result);

	// Collect final detections after NMS
	for (int i = 0; i < NMS_result.size(); ++i) {
		Detection result;
		const unsigned short id = NMS_result[i];

		result.class_id = class_list[id];
		result.confidence = confidence_list[id];
		result.box = GetBoundingBox(box_list[id]);

		cropped_face = DrawDetectedObject(frame, result);
		last_box = result.box;
	}
	return cropped_face;
}

// Method to get the bounding box in the correct scale
cv::Rect Inference::GetBoundingBox(const cv::Rect &src) const {
	cv::Rect box = src;
	box.x = (box.x - box.width / 2) * scale_factor_.x;
	box.y = (box.y - box.height / 2) * scale_factor_.y;
	box.width *= scale_factor_.x;
	box.height *= scale_factor_.y;
	return box;
}

cv:: Mat Inference::DrawDetectedObject(cv::Mat &frame, const Detection &detection) const {
	const cv::Rect &box = detection.box;
	const float &confidence = detection.confidence;
	const int &class_id = detection.class_id;
	
	// Generate a random color for the bounding box
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> dis(120, 255);
	const cv::Scalar &color = cv::Scalar(dis(gen), dis(gen), dis(gen));
	
	// Define the rectangle (ROI)
    cv::Rect roi(box.x, box.y, box.width, box.height);
	// Crop the image using the bounding box
    cv::Mat cropped_image = frame(roi);
	
	// // Draw the bounding box around the detected object
	// cv::rectangle(frame, cv::Point(box.x, box.y), cv::Point(box.x + box.width, box.y + box.height), color, 3);
	// // Prepare the class label and confidence text
	// std::string classString = classes_[class_id] + std::to_string(confidence).substr(0, 4);
	
	// // Get the size of the text box
	// cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 0.75, 2, 0);
	// cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);
	
	// // Draw the text box
	// cv::rectangle(frame, textBox, color, cv::FILLED);
	
	// // Put the class label and confidence text above the bounding box
	// cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 0.75, cv::Scalar(0, 0, 0), 2, 0);
	return cropped_image;
}
} // namespace yolo