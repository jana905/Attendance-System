#include "face_embedding.hpp"
#include <stdexcept>
#include <filesystem>
#include <iostream>
#include <fstream>




FaceEmbeddingIntegration::FaceEmbeddingIntegration(
    const cv::Mat &input_image,
    ModelSelection model_selection,
    const ModelPaths &paths,
    const std::string &device,
    std::shared_ptr<ModelDecryptor> decryptor)
    : input_image_(input_image.clone()),
      model_selection_(model_selection),
      model_paths_(paths),
      device_(device),
      decryptor_(decryptor)
{
    if (input_image_.empty()) {
        throw std::invalid_argument("Input image is empty");
    }

    initializeSelectedModels();
}




void FaceEmbeddingIntegration::initializeSelectedModels() {
    try {
        switch (model_selection_) {
        case ModelSelection::ARCFACE_ONLY:
            arcface_ = std::make_unique<ArcFaceInference>(
                model_paths_.arcface_path,
                decryptor_,
                device_
            );
            break;

        case ModelSelection::FACENET_ONLY:
            facenet_ = std::make_unique<FaceNetInference>(
                model_paths_.facenet_path,
                decryptor_,
                device_
            );
            break;

        case ModelSelection::BOTH_MODELS:
            arcface_ = std::make_unique<ArcFaceInference>(
                model_paths_.arcface_path,
                decryptor_,
                device_
            );
            facenet_ = std::make_unique<FaceNetInference>(
                model_paths_.facenet_path,
                decryptor_,
                device_
            );
            break;

        default:
            throw std::invalid_argument("Invalid model selection");
        }
    }
    catch (const std::exception &e) {
        throw std::runtime_error("Model initialization failed: " + std::string(e.what()));
    }
}





std::vector<float> FaceEmbeddingIntegration::runInference()
{
    embeddings_.clear();

    try
    {
        switch (model_selection_)
        {
        case ModelSelection::ARCFACE_ONLY:
            arcface_embeddings_ = arcface_->run(input_image_);
            break;

        case ModelSelection::FACENET_ONLY:
            facenet_embeddings_ = facenet_->run(input_image_);
            break;

        case ModelSelection::BOTH_MODELS:
            arcface_embeddings_ = arcface_->run(input_image_);
            facenet_embeddings_ = facenet_->run(input_image_);
            break;

        default:
            throw std::invalid_argument("Invalid model selection for inference");
        }

        // Combine embeddings
        if (!arcface_embeddings_.empty())
        {
            embeddings_.insert(embeddings_.end(),
                               arcface_embeddings_.ptr<float>(),
                               arcface_embeddings_.ptr<float>() + arcface_embeddings_.total());
        }
        if (!facenet_embeddings_.empty())
        {
            embeddings_.insert(embeddings_.end(),
                               facenet_embeddings_.ptr<float>(),
                               facenet_embeddings_.ptr<float>() + facenet_embeddings_.total());
        }

        return embeddings_;
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error("Inference failed: " + std::string(e.what()));
    }
}

void FaceEmbeddingIntegration::saveTensorOutput(
    const cv::Mat &embeddings,
    const std::string &filepath) const
{
    try
    {
        std::filesystem::create_directories(std::filesystem::path(filepath).parent_path());

        std::ofstream output_file(filepath, std::ios::binary);
        if (!output_file.is_open())
        {
            throw std::runtime_error("Could not open file for writing: " + filepath);
        }

        // Write tensor metadata
        int dims = 2;
        int sizes[2] = {1, embeddings.cols};
        int dtype_int = 21; // CV_32FC1

        output_file.write(reinterpret_cast<const char *>(&dims), sizeof(dims));
        output_file.write(reinterpret_cast<const char *>(sizes), sizeof(sizes));
        output_file.write(reinterpret_cast<const char *>(&dtype_int), sizeof(dtype_int));

        // Write embedding data
        output_file.write(reinterpret_cast<const char *>(embeddings.data),
                          embeddings.total() * sizeof(float));

        // std::cout << "Embeddings saved to " << filepath << std::endl;
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error("Failed to save embeddings: " + std::string(e.what()));
    }
}

bool FaceEmbeddingIntegration::isUsingArcFace() const
{
    return model_selection_ == ModelSelection::ARCFACE_ONLY ||
           model_selection_ == ModelSelection::BOTH_MODELS;
}

bool FaceEmbeddingIntegration::isUsingFaceNet() const
{
    return model_selection_ == ModelSelection::FACENET_ONLY ||
           model_selection_ == ModelSelection::BOTH_MODELS;
}