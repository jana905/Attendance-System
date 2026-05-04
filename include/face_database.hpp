#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <filesystem>
#include <algorithm>
#include <opencv2/opencv.hpp>

// ── Model selection ───────────────────────────────────────────────────────────
#include "model_selection.hpp"

// ── Single candidate in Top-K results ────────────────────────────────────────
struct Candidate {
    std::string person_id;
    std::string person_name;
    float arcface_similarity  = -1.0f;
    float facenet_similarity  = -1.0f;
    float buffalo_similarity  = -1.0f;
    float ghost_similarity    = -1.0f;
    float combined_similarity = -1.0f;  // weighted average of active models
    int   rank                = 0;      // 1 = best match
};

// ── Best match result (top-1 + metadata) ─────────────────────────────────────
struct MatchResult {
    bool matched               = false;
    std::string person_id;
    std::string person_name;
    float arcface_similarity   = -1.0f;
    float facenet_similarity   = -1.0f;
    float buffalo_similarity   = -1.0f;
    float ghost_similarity     = -1.0f;
    float combined_similarity  = -1.0f;
    ModelSelection model_used  = ModelSelection::ARCFACE_ONLY;
    double inference_time_ms   = 0.0;

    // Full ranked list (populated when top_k > 1)
    std::vector<Candidate> top_k;
};

// ── Per-person DB record ──────────────────────────────────────────────────────
struct PersonRecord {
    std::string name;
    std::string id;
    std::string enrolled_at;
    cv::Mat arcface_emb;
    cv::Mat facenet_emb;
    cv::Mat buffalo_emb;
    cv::Mat ghost_emb;
};

class FaceDatabase {
public:
    explicit FaceDatabase(const std::string& db_dir = "db");
    ~FaceDatabase() = default;

    // Enroll — pass empty Mat for models not used
    std::string enroll(
        const std::string& name,
        const cv::Mat& arcface_emb,
        const cv::Mat& facenet_emb,
        const cv::Mat& buffalo_emb,
        const cv::Mat& ghost_emb);

    // Top-K search — returns MatchResult with top_k populated
    MatchResult findTopK(
        const cv::Mat& arcface_emb,
        const cv::Mat& facenet_emb,
        const cv::Mat& buffalo_emb,
        const cv::Mat& ghost_emb,
        ModelSelection model,
        float threshold = 0.35f,
        int   k         = 5) const;

    // Convenience: top-1 only (backward compatible)
    MatchResult findMatch(
        const cv::Mat& arcface_emb,
        const cv::Mat& facenet_emb,
        const cv::Mat& buffalo_emb,
        const cv::Mat& ghost_emb,
        ModelSelection model,
        float threshold = 0.35f) const;

    void load();
    void save() const;

    size_t size()  const { return records_.size(); }
    bool   empty() const { return records_.empty(); }
    std::vector<std::string> listNames() const;
    bool remove(const std::string& person_id);

private:
    std::string db_dir_;
    std::string index_path_;
    std::unordered_map<std::string, PersonRecord> records_;

    static float cosineSimilarity(const cv::Mat& a, const cv::Mat& b);
    static std::string generateId();
    static std::string currentTimestamp();

    void    saveEmbedding(const cv::Mat& emb, const std::string& path) const;
    cv::Mat loadEmbedding(const std::string& path) const;
    void writeIndex() const;
    void readIndex();
};
