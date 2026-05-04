#ifndef FACE_EMBEDDING_HPP
#define FACE_EMBEDDING_HPP

#include "arcface.hpp"
#include "facenet.hpp"
#include <vector>
#include <memory>
#include <string>
#include <opencv2/opencv.hpp>
#include "model_decryptor.hpp"

enum class ModelSelection
{
    ARCFACE_ONLY,
    FACENET_ONLY,
    BOTH_MODELS
};

class FaceEmbeddingIntegration

{
public:

    struct ModelPaths
    {
        std::string arcface_path;
        std::string facenet_path;

        ModelPaths(
            std::string arcface = "models/arcface/arcface_openvino.xml",
            std::string facenet = "models/facenet/facenet512.xml") : arcface_path(arcface), facenet_path(facenet) {}
    };




    // Constructor declaration

    FaceEmbeddingIntegration(
        const cv::Mat &input_image,
        ModelSelection model_selection,
        const ModelPaths &paths = ModelPaths(),
        const std::string &device = "AUTO");


    FaceEmbeddingIntegration(
        const cv::Mat& input_image,
        ModelSelection model_selection,
        const ModelPaths& paths = ModelPaths(),
        const std::string& device = "AUTO",
        std::shared_ptr<ModelDecryptor> decryptor = nullptr);


    ~FaceEmbeddingIntegration() {
        // Cleanup any temporary files
        for (const auto& path : temp_files_) {
            std::filesystem::remove(path);
        }
    }
    std::vector<float> runInference();
    void saveTensorOutput(const cv::Mat &embeddings, const std::string &filepath) const;

    cv::Mat getArcFaceEmbeddings() const { return arcface_embeddings_; }
    cv::Mat getFaceNetEmbeddings() const { return facenet_embeddings_; }

    bool isUsingArcFace() const;
    bool isUsingFaceNet() const;

private:
  // Add decryptor and temporary file tracking
    cv::Mat input_image_;
    ModelSelection model_selection_;
    ModelPaths model_paths_;
    std::string device_;
    std::shared_ptr<ModelDecryptor> decryptor_;

    std::vector<std::string> temp_files_;
    



    std::unique_ptr<ArcFaceInference> arcface_;
    std::unique_ptr<FaceNetInference> facenet_;


    std::vector<float> embeddings_;
    cv::Mat arcface_embeddings_;
    cv::Mat facenet_embeddings_;

    void initializeSelectedModels();
};

#endif // FACE_EMBEDDING_HPP