#pragma once

#include <string>
#include <memory>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "inference.hpp"
#include "face_embedding.hpp"
#include "face_database.hpp"
#include "attendance_logger.hpp"
#include "model_decryptor.hpp"
#include "arcface.hpp"
#include "facenet.hpp"

struct FrameStats {
    double detection_ms   = 0.0;
    double arcface_ms     = 0.0;
    double facenet_ms     = 0.0;
    double buffalo_ms     = 0.0;
    double ghost_ms       = 0.0;
    double total_embed_ms = 0.0;
    double match_ms       = 0.0;
    double total_ms       = 0.0;
    bool   face_detected  = false;
    bool   matched        = false;
};

struct BenchmarkReport {
    size_t total_frames     = 0;
    size_t frames_with_face = 0;
    size_t frames_matched   = 0;

    double avg_detection_ms   = 0.0;
    double avg_arcface_ms     = 0.0;
    double avg_facenet_ms     = 0.0;
    double avg_buffalo_ms     = 0.0;
    double avg_ghost_ms       = 0.0;
    double avg_total_embed_ms = 0.0;
    double avg_match_ms       = 0.0;
    double avg_total_ms       = 0.0;
    double min_total_ms       = 1e9;
    double max_total_ms       = 0.0;

    ModelSelection model;
    std::string    model_name;

    void print() const;
    void saveCsv(const std::string& path) const;
};

class Recognizer {
public:
    struct Config {
        std::string yolo_model_path;
        std::string arcface_model_path;
        std::string facenet_model_path;
        std::string buffalo_model_path;
        std::string ghost_model_path;
        std::string key_path           = "";
        std::string device             = "AUTO";
        std::string db_dir             = "db";
        std::string attendance_dir     = "attendance";
        std::string unknown_dir        = "unknown_faces"; // saved crops for review
        float       det_confidence     = 0.5f;
        float       det_nms            = 0.5f;
        float       recognition_thresh = 0.35f;
        int         cooldown_seconds   = 60;
        int         top_k              = 1;
        int         unknown_cooldown   = 10; // min seconds between saving same unknown
        ModelSelection model_selection = ModelSelection::ARCFACE_ONLY;
    };

    explicit Recognizer(const Config& cfg);
    ~Recognizer() = default;

    std::string     enroll(const std::string& name, const cv::Mat& image);
    // Enroll all subfolders in faces_dir — folder name = person name
    // Returns number of people enrolled. Skips folders with no valid faces.
    int             enrollDatabase(const std::string& faces_dir,
                                   bool re_enroll = false);

    std::string     enrollFolder(const std::string& name,
                                 const std::string& folder_path,
                                 int* images_used = nullptr);
    MatchResult     processFrame(cv::Mat& frame, FrameStats* stats = nullptr);
    BenchmarkReport benchmark(int camera_id, int num_frames = 300);

    static void runFullBenchmark(const Config& base_cfg,
                                 int camera_id, int num_frames,
                                 const std::string& output_csv = "benchmark_results.csv");

    FaceDatabase&     database() { return *db_; }
    AttendanceLogger& logger()   { return *logger_; }

private:
    Config cfg_;
    std::shared_ptr<ModelDecryptor>    decryptor_;
    std::unique_ptr<yolo::Inference>   detector_;
    std::shared_ptr<FaceDatabase>      db_;
    std::shared_ptr<AttendanceLogger>  logger_;

    std::unique_ptr<ArcFaceInference>  arcface_engine_;
    std::unique_ptr<FaceNetInference>  facenet_engine_;
    ov::CompiledModel buffalo_compiled_;
    ov::CompiledModel ghost_compiled_;
    bool buffalo_loaded_ = false;
    bool ghost_loaded_   = false;

    // Unknown face cooldown tracking
    std::chrono::steady_clock::time_point last_unknown_saved_;
    bool unknown_ever_saved_ = false;
    int  frames_processed_   = 0;   // skip unknown saves during warm-up

    struct Embeddings { cv::Mat arcface, facenet, buffalo, ghost; };
    Embeddings generateEmbeddings(const cv::Mat& face_crop, FrameStats* stats = nullptr);

    cv::Mat runBuffalo(const cv::Mat& face_crop);
    cv::Mat runGhost(const cv::Mat& face_crop);

    // Draw bbox + labels on frame. bbox is the face rect in full-frame coords.
    void drawOverlay(cv::Mat& frame, const MatchResult& result,
                     const FrameStats& stats, const cv::Rect& bbox) const;

    // Save cropped face to unknown_dir with timestamp filename.
    void saveUnknownFace(const cv::Mat& face_crop);

    static std::string modelLabel(ModelSelection m);
    static std::string timestampFilename();
};
