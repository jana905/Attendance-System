#include "face_database.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <random>
#include <stdexcept>
#include <iostream>

FaceDatabase::FaceDatabase(const std::string& db_dir)
    : db_dir_(db_dir), index_path_(db_dir + "/index.csv")
{
    std::filesystem::create_directories(db_dir_ + "/embeddings");
    load();
}

std::string FaceDatabase::enroll(
    const std::string& name,
    const cv::Mat& arcface_emb,
    const cv::Mat& facenet_emb,
    const cv::Mat& buffalo_emb,
    const cv::Mat& ghost_emb)
{
    if (name.empty())
        throw std::invalid_argument("Person name cannot be empty");
    if (arcface_emb.empty() && facenet_emb.empty() &&
        buffalo_emb.empty() && ghost_emb.empty())
        throw std::invalid_argument("At least one embedding must be provided");

    PersonRecord rec;
    rec.id          = generateId();
    rec.name        = name;
    rec.enrolled_at = currentTimestamp();

    auto save = [&](const cv::Mat& emb, const std::string& tag, cv::Mat& dest) {
        if (!emb.empty()) {
            std::string p = db_dir_ + "/embeddings/" + rec.id + "_" + tag + ".bin";
            saveEmbedding(emb, p);
            dest = emb.clone();
        }
    };
    save(arcface_emb, "arcface", rec.arcface_emb);
    save(facenet_emb, "facenet", rec.facenet_emb);
    save(buffalo_emb, "buffalo", rec.buffalo_emb);
    save(ghost_emb,   "ghost",   rec.ghost_emb);

    std::string new_id = rec.id;
    records_[new_id] = std::move(rec);
    writeIndex();
    std::cout << "[DB] Enrolled '" << name << "' – ID: " << new_id << "\n";
    return new_id;
}

// ── Top-K search ──────────────────────────────────────────────────────────────

MatchResult FaceDatabase::findTopK(
    const cv::Mat& arcface_emb,
    const cv::Mat& facenet_emb,
    const cv::Mat& buffalo_emb,
    const cv::Mat& ghost_emb,
    ModelSelection model,
    float threshold,
    int   k) const
{
    bool use_af = (model == ModelSelection::ARCFACE_ONLY ||
                   model == ModelSelection::BOTH_MODELS  ||
                   model == ModelSelection::ALL_MODELS);
    bool use_fn = (model == ModelSelection::FACENET_ONLY ||
                   model == ModelSelection::BOTH_MODELS  ||
                   model == ModelSelection::ALL_MODELS);
    bool use_bu = (model == ModelSelection::BUFFALO_L    ||
                   model == ModelSelection::ALL_MODELS);
    bool use_gh = (model == ModelSelection::GHOSTFACENET ||
                   model == ModelSelection::ALL_MODELS);

    // Score all records
    std::vector<Candidate> candidates;
    candidates.reserve(records_.size());

    for (const auto& [id, rec] : records_) {
        float score = 0.0f;
        int   count = 0;
        float af=-1, fn=-1, bu=-1, gh=-1;

        auto tryMatch = [&](const cv::Mat& q, const cv::Mat& db,
                            float& out, bool use) {
            if (use && !q.empty() && !db.empty()) {
                out    = cosineSimilarity(q, db);
                score += out;
                ++count;
            }
        };

        tryMatch(arcface_emb, rec.arcface_emb, af, use_af);
        tryMatch(facenet_emb, rec.facenet_emb, fn, use_fn);
        tryMatch(buffalo_emb, rec.buffalo_emb, bu, use_bu);
        tryMatch(ghost_emb,   rec.ghost_emb,   gh, use_gh);

        if (count == 0) continue;

        Candidate c;
        c.person_id           = id;
        c.person_name         = rec.name;
        c.arcface_similarity  = af;
        c.facenet_similarity  = fn;
        c.buffalo_similarity  = bu;
        c.ghost_similarity    = gh;
        c.combined_similarity = score / static_cast<float>(count);
        candidates.push_back(c);
    }

    // Sort descending by combined similarity
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.combined_similarity > b.combined_similarity;
              });

    // Assign ranks
    for (int i = 0; i < (int)candidates.size(); ++i)
        candidates[i].rank = i + 1;

    // Keep top-k
    if ((int)candidates.size() > k)
        candidates.resize(k);

    // Build result
    MatchResult result;
    result.model_used = model;
    result.top_k      = candidates;

    if (!candidates.empty()) {
        const auto& best      = candidates[0];
        result.person_id      = best.person_id;
        result.person_name    = best.person_name;
        result.arcface_similarity  = best.arcface_similarity;
        result.facenet_similarity  = best.facenet_similarity;
        result.buffalo_similarity  = best.buffalo_similarity;
        result.ghost_similarity    = best.ghost_similarity;
        result.combined_similarity = best.combined_similarity;
        result.matched             = best.combined_similarity >= threshold;
    }

    return result;
}

MatchResult FaceDatabase::findMatch(
    const cv::Mat& arcface_emb,
    const cv::Mat& facenet_emb,
    const cv::Mat& buffalo_emb,
    const cv::Mat& ghost_emb,
    ModelSelection model,
    float threshold) const
{
    return findTopK(arcface_emb, facenet_emb, buffalo_emb, ghost_emb,
                    model, threshold, 1);
}

void FaceDatabase::load() {
    records_.clear();
    if (!std::filesystem::exists(index_path_)) return;
    readIndex();
}

void FaceDatabase::save() const { writeIndex(); }

std::vector<std::string> FaceDatabase::listNames() const {
    std::vector<std::string> names;
    for (const auto& [id, rec] : records_)
        names.push_back(rec.name);
    return names;
}

bool FaceDatabase::remove(const std::string& person_id) {
    auto it = records_.find(person_id);
    if (it == records_.end()) return false;
    for (const auto& tag : {"arcface","facenet","buffalo","ghost"})
        std::filesystem::remove(db_dir_+"/embeddings/"+person_id+"_"+tag+".bin");
    records_.erase(it);
    writeIndex();
    return true;
}

float FaceDatabase::cosineSimilarity(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat fa = a.reshape(1,1);
    cv::Mat fb = b.reshape(1,1);
    if (fa.cols != fb.cols)
        throw std::runtime_error("Embedding dim mismatch");
    double dot    = fa.dot(fb);
    double norm_a = cv::norm(fa);
    double norm_b = cv::norm(fb);
    if (norm_a < 1e-8 || norm_b < 1e-8) return 0.0f;
    return static_cast<float>(dot / (norm_a * norm_b));
}

std::string FaceDatabase::generateId() {
    auto now = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937_64 rng(now ^ std::random_device{}());
    std::ostringstream oss;
    oss << std::hex << std::setw(12) << std::setfill('0') << rng();
    return oss.str().substr(0, 12);
}

std::string FaceDatabase::currentTimestamp() {
    auto now  = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

void FaceDatabase::saveEmbedding(const cv::Mat& emb, const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write: " + path);
    int cols = emb.cols;
    f.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    f.write(reinterpret_cast<const char*>(emb.ptr<float>(0)), cols*sizeof(float));
}

cv::Mat FaceDatabase::loadEmbedding(const std::string& path) const {
    if (!std::filesystem::exists(path)) return cv::Mat();
    std::ifstream f(path, std::ios::binary);
    if (!f) return cv::Mat();
    int cols = 0;
    f.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    cv::Mat emb(1, cols, CV_32F);
    f.read(reinterpret_cast<char*>(emb.ptr<float>(0)), cols*sizeof(float));
    return emb;
}

void FaceDatabase::writeIndex() const {
    std::ofstream f(index_path_);
    if (!f) throw std::runtime_error("Cannot write index");
    f << "id,name,enrolled_at,has_arcface,has_facenet,has_buffalo,has_ghost\n";
    for (const auto& [id, rec] : records_) {
        f << rec.id << "," << rec.name << "," << rec.enrolled_at << ","
          << (!rec.arcface_emb.empty()?"1":"0") << ","
          << (!rec.facenet_emb.empty()?"1":"0") << ","
          << (!rec.buffalo_emb.empty()?"1":"0") << ","
          << (!rec.ghost_emb.empty()  ?"1":"0") << "\n";
    }
}

void FaceDatabase::readIndex() {
    std::ifstream f(index_path_);
    if (!f) return;
    std::string line;
    std::getline(f, line); // header
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string id,name,enrolled_at,af,fn,bu,gh;
        std::getline(ss,id,','); std::getline(ss,name,',');
        std::getline(ss,enrolled_at,','); std::getline(ss,af,',');
        std::getline(ss,fn,','); std::getline(ss,bu,',');
        std::getline(ss,gh,',');
        PersonRecord rec;
        rec.id=id; rec.name=name; rec.enrolled_at=enrolled_at;
        auto load = [&](const std::string& flag, const std::string& tag) -> cv::Mat {
            return flag=="1"
                ? loadEmbedding(db_dir_+"/embeddings/"+id+"_"+tag+".bin")
                : cv::Mat();
        };
        rec.arcface_emb = load(af,"arcface");
        rec.facenet_emb = load(fn,"facenet");
        rec.buffalo_emb = load(bu,"buffalo");
        rec.ghost_emb   = load(gh,"ghost");
        records_[id] = std::move(rec);
    }
    std::cout << "[DB] Loaded " << records_.size() << " records.\n";
}
