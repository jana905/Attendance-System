#ifndef YOLO_INFERENCE_H_
#define YOLO_INFERENCE_H_

#include <string>
#include <vector>
#include <memory>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>
#include "model_decryptor.hpp"
#include <filesystem>


namespace yolo {

struct Detection {
	short class_id;
	float confidence;
	cv::Rect box;
};



class Inference {
 public:
     // Add decryptor pointer to store reference

	cv::Mat cropped_face;
	cv::Rect last_box;



    // Constructor with custom input shape and decryption
    Inference(const std::string &model_path,
             const cv::Size model_input_shape,
             const float &model_confidence_threshold,
             const float &model_NMS_threshold,
             std::shared_ptr<ModelDecryptor> decryptor = nullptr);



   // Add destructor for cleanup
    ~Inference() {
        // Cleanup temporary files
        for (const auto& path : temp_files_) {
            // std::filesystem::remove(path);
		std::filesystem::remove(std::filesystem::path(path));

        }
    }

	void RunInference(cv::Mat &frame);



 private:
	void InitializeModel(const std::string &model_path);
	void Preprocessing(const cv::Mat &frame);
	cv::Mat PostProcessing(cv::Mat &frame);
	cv::Rect GetBoundingBox(const cv::Rect &src) const;
	cv::Mat DrawDetectedObject(cv::Mat &frame, const Detection &detections) const;

	cv::Point2f scale_factor_;			// Scaling factor for the input frame
	cv::Size2f model_input_shape_;	// Input shape of the model
	cv::Size model_output_shape_;		// Output shape of the model

	ov::InferRequest inference_request_;  // OpenVINO inference request
	ov::CompiledModel compiled_model_;    // OpenVINO compiled model


    // Decryption support
    std::shared_ptr<ModelDecryptor> decryptor_;
    std::vector<std::string> temp_files_;

	float model_confidence_threshold_;  // Confidence threshold for detections
	float model_NMS_threshold_;         // Non-Maximum Suppression threshold

	std::vector<std::string> classes_ {
		"face"
	};
};

} // namespace yolo

#endif // YOLO_INFERENCE_H_