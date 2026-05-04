#include "recognizer.hpp"
#include "attendance_logger.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <stdexcept>
#include <filesystem>

using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::duration<double, std::milli>;

// ── BenchmarkReport ──────────────────────────────────────────────────────────

void BenchmarkReport::print() const {
    auto row = [](const std::string& lbl, double v) {
        std::cout << "  " << std::left << std::setw(24) << lbl
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(8) << v << " ms\n";
    };
    std::cout << "\n┌─ " << model_name << " ─────────────────────────────┐\n"
              << "  frames=" << total_frames
              << "  with_face=" << frames_with_face
              << "  matched=" << frames_matched << "\n"
              << "  ──────────────────────────────────────────────\n";
    row("Avg detection",       avg_detection_ms);
    row("Avg ArcFace embed",   avg_arcface_ms);
    row("Avg FaceNet embed",   avg_facenet_ms);
    row("Avg Buffalo_L embed", avg_buffalo_ms);
    row("Avg GhostFaceNet",    avg_ghost_ms);
    row("Avg total embed",     avg_total_embed_ms);
    row("Avg DB match",        avg_match_ms);
    row("Avg E2E",             avg_total_ms);
    row("Min E2E",             min_total_ms==1e9?0.0:min_total_ms);
    row("Max E2E",             max_total_ms);
    std::cout << "└────────────────────────────────────────────────────┘\n\n";
}

void BenchmarkReport::saveCsv(const std::string& path) const {
    bool is_new = !std::filesystem::exists(path);
    std::ofstream f(path, std::ios::app);
    if (!f) return;
    if (is_new)
        f << "model,total_frames,with_face,matched,"
             "avg_det_ms,avg_arcface_ms,avg_facenet_ms,"
             "avg_buffalo_ms,avg_ghost_ms,"
             "avg_embed_ms,avg_match_ms,avg_total_ms,"
             "min_ms,max_ms\n";
    f << model_name << ","
      << total_frames << "," << frames_with_face << "," << frames_matched << ","
      << std::fixed << std::setprecision(3)
      << avg_detection_ms << "," << avg_arcface_ms << "," << avg_facenet_ms << ","
      << avg_buffalo_ms   << "," << avg_ghost_ms   << ","
      << avg_total_embed_ms << "," << avg_match_ms << "," << avg_total_ms << ","
      << (min_total_ms==1e9?0.0:min_total_ms) << "," << max_total_ms << "\n";
}

// ── Construction ─────────────────────────────────────────────────────────────

Recognizer::Recognizer(const Config& cfg) : cfg_(cfg) {
    std::filesystem::create_directories(cfg_.unknown_dir);

    decryptor_ = std::make_shared<ModelDecryptor>(cfg_.key_path);
    detector_  = std::make_unique<yolo::Inference>(
        cfg_.yolo_model_path, cv::Size(640,640),
        cfg_.det_confidence, cfg_.det_nms, decryptor_);

    db_     = std::make_shared<FaceDatabase>(cfg_.db_dir);
    logger_ = std::make_shared<AttendanceLogger>(cfg_.attendance_dir,
                                                  cfg_.cooldown_seconds);
    ModelSelection m = cfg_.model_selection;

    bool need_af = (m==ModelSelection::ARCFACE_ONLY||m==ModelSelection::BOTH_MODELS||m==ModelSelection::ALL_MODELS);
    bool need_fn = (m==ModelSelection::FACENET_ONLY||m==ModelSelection::BOTH_MODELS||m==ModelSelection::ALL_MODELS);
    bool need_bu = (m==ModelSelection::BUFFALO_L   ||m==ModelSelection::ALL_MODELS);
    bool need_gh = (m==ModelSelection::GHOSTFACENET||m==ModelSelection::ALL_MODELS);

    if (need_af && !cfg_.arcface_model_path.empty() &&
        std::filesystem::exists(cfg_.arcface_model_path)) {
        std::cout << "[Recognizer] Loading ArcFace...\n";
        arcface_engine_ = std::make_unique<ArcFaceInference>(
            cfg_.arcface_model_path, nullptr, cfg_.device);
    }
    if (need_fn && !cfg_.facenet_model_path.empty() &&
        std::filesystem::exists(cfg_.facenet_model_path)) {
        std::cout << "[Recognizer] Loading FaceNet512...\n";
        facenet_engine_ = std::make_unique<FaceNetInference>(
            cfg_.facenet_model_path, nullptr, cfg_.device);
    }
    if (need_bu && !cfg_.buffalo_model_path.empty() &&
        std::filesystem::exists(cfg_.buffalo_model_path)) {
        std::cout << "[Recognizer] Loading Buffalo_L...\n";
        ov::Core core;
        buffalo_compiled_ = core.compile_model(
            core.read_model(cfg_.buffalo_model_path), cfg_.device);
        buffalo_loaded_ = true;
    }
    if (need_gh && !cfg_.ghost_model_path.empty() &&
        std::filesystem::exists(cfg_.ghost_model_path)) {
        std::cout << "[Recognizer] Loading GhostFaceNet...\n";
        ov::Core core;
        ghost_compiled_ = core.compile_model(
            core.read_model(cfg_.ghost_model_path), cfg_.device);
        ghost_loaded_ = true;
    }

    std::cout << "[Recognizer] Ready. Model=" << modelLabel(m)
              << "  DB=" << db_->size() << " people"
              << "  TopK=" << cfg_.top_k << "\n";
}

// ── Enrollment ───────────────────────────────────────────────────────────────

std::string Recognizer::enroll(const std::string& name, const cv::Mat& image) {
    cv::Mat frame = image.clone();
    detector_->RunInference(frame);
    if (detector_->cropped_face.empty())
        throw std::runtime_error("No face detected for: " + name);
    auto embs = generateEmbeddings(detector_->cropped_face);
    return db_->enroll(name, embs.arcface, embs.facenet, embs.buffalo, embs.ghost);
}

// ── Frame processing ─────────────────────────────────────────────────────────

MatchResult Recognizer::processFrame(cv::Mat& frame, FrameStats* stats_out) {
    FrameStats stats;
    MatchResult result;
    result.model_used = cfg_.model_selection;
    auto t0 = Clock::now();

    auto td = Clock::now();
    detector_->RunInference(frame);
    stats.detection_ms = Ms(Clock::now()-td).count();

    if (detector_->cropped_face.empty()) {
        stats.total_ms = Ms(Clock::now()-t0).count();
        if (stats_out) *stats_out = stats;
        return result;
    }
    stats.face_detected = true;

    // Retrieve bbox — populated by RunInference into last_box
    cv::Rect bbox = detector_->last_box;
    bbox &= cv::Rect(0, 0, frame.cols, frame.rows);

    auto embs = generateEmbeddings(detector_->cropped_face, &stats);

    auto tm = Clock::now();
    if (!db_->empty())
        result = db_->findTopK(
            embs.arcface, embs.facenet, embs.buffalo, embs.ghost,
            cfg_.model_selection, cfg_.recognition_thresh, cfg_.top_k);
    stats.match_ms = Ms(Clock::now()-tm).count();
    stats.matched  = result.matched;
    stats.total_ms = Ms(Clock::now()-t0).count();
    result.inference_time_ms = stats.total_embed_ms;

    ++frames_processed_;

    if (result.matched) {
        logger_->log(result, stats.total_ms);
        // Do NOT save as unknown — person is recognized, logger may just be in cooldown
    } else if (stats.face_detected && frames_processed_ > 5) {
        // Skip first 5 frames to allow model warm-up before flagging unknowns
        saveUnknownFace(detector_->cropped_face);
    }

    drawOverlay(frame, result, stats, bbox);

    if (stats_out) *stats_out = stats;
    return result;
}

// ── Draw overlay ─────────────────────────────────────────────────────────────

void Recognizer::drawOverlay(cv::Mat& frame, const MatchResult& result,
                              const FrameStats& stats, const cv::Rect& bbox) const
{
    // Bounding box
    if (bbox.width > 0 && bbox.height > 0) {
        cv::Scalar box_color = result.matched
            ? cv::Scalar(0, 220, 0)
            : cv::Scalar(0, 0, 220);
        cv::rectangle(frame, bbox, box_color, 2);

        // Label pinned above box
        if (!result.top_k.empty()) {
            const auto& best = result.top_k[0];
            std::ostringstream lbl;
            if (result.matched) {
                lbl << best.person_name << " "
                    << std::fixed << std::setprecision(0)
                    << best.combined_similarity * 100 << "%";
            } else {
                lbl << "Unknown";
                if (best.combined_similarity > 0)
                    lbl << " (best:" << best.person_name << " "
                        << std::fixed << std::setprecision(0)
                        << best.combined_similarity * 100 << "%)";
            }

            int baseline = 0;
            cv::Size tsz = cv::getTextSize(lbl.str(),
                cv::FONT_HERSHEY_SIMPLEX, 0.65, 2, &baseline);
            cv::Point org(bbox.x, std::max(bbox.y - 8, tsz.height + 4));
            cv::rectangle(frame,
                cv::Point(org.x - 2, org.y - tsz.height - 2),
                cv::Point(org.x + tsz.width + 2, org.y + baseline),
                box_color, cv::FILLED);
            cv::putText(frame, lbl.str(), org,
                cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255,255,255), 2);
        }
    }

    // Top-K list (top-left)
    int y = 30;
    if (result.top_k.empty()) {
        cv::putText(frame, "No face", cv::Point(10, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(120,120,120), 2);
    } else {
        for (int i = 0; i < (int)result.top_k.size(); ++i) {
            const auto& c = result.top_k[i];
            cv::Scalar color = (i==0 && result.matched)
                ? cv::Scalar(0,220,0)
                : (i==0 ? cv::Scalar(0,80,220) : cv::Scalar(180,180,0));
            std::ostringstream row;
            row << "#" << (i+1) << " " << c.person_name
                << "  " << std::fixed << std::setprecision(0)
                << c.combined_similarity*100 << "%";
            if (i==0 && result.matched) row << "  ✔";
            cv::putText(frame, row.str(), cv::Point(10, y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.60, color, 2);
            y += 26;
        }
    }

    // Unknown badge
    if (stats.face_detected && !result.matched) {
        cv::putText(frame, "[ UNKNOWN - saved for review ]",
                    cv::Point(10, y + 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0,80,220), 2);
    }

    // Perf bar
    std::ostringstream perf;
    perf << std::fixed << std::setprecision(1)
         << "E2E:" << stats.total_ms << "ms"
         << "  FPS:" << (stats.total_ms>0 ? 1000.0/stats.total_ms : 0.0);
    cv::putText(frame, perf.str(), cv::Point(10, frame.rows-10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200,200,0), 1);

    // Model tag
    cv::putText(frame,
                "["+modelLabel(cfg_.model_selection)+"|K="+
                std::to_string(cfg_.top_k)+"]",
                cv::Point(frame.cols-220, 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(200,200,0), 1);
}

// ── Save unknown face ─────────────────────────────────────────────────────────

void Recognizer::saveUnknownFace(const cv::Mat& face_crop) {
    if (face_crop.empty()) return;

    auto now = Clock::now();
    if (unknown_ever_saved_) {
        double elapsed = std::chrono::duration<double>(
            now - last_unknown_saved_).count();
        if (elapsed < static_cast<double>(cfg_.unknown_cooldown))
            return;
    }

    std::string path = cfg_.unknown_dir + "/unknown_" + timestampFilename() + ".jpg";
    if (cv::imwrite(path, face_crop)) {
        std::cout << "[Unknown] Saved: " << path << "\n";
        last_unknown_saved_ = now;
        unknown_ever_saved_ = true;
    } else {
        std::cerr << "[Unknown] Failed to write: " << path << "\n";
    }
}

std::string Recognizer::timestampFilename() {
    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(
                   now.time_since_epoch()) % 1000;
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&t), "%Y%m%d_%H%M%S")
        << "_" << std::setw(3) << std::setfill('0') << ms.count();
    return oss.str();
}

// ── Benchmark ────────────────────────────────────────────────────────────────

BenchmarkReport Recognizer::benchmark(int camera_id, int num_frames) {
    cv::VideoCapture cap(camera_id);
    if (!cap.isOpened())
        throw std::runtime_error("Cannot open camera "+std::to_string(camera_id));

    BenchmarkReport report;
    report.model      = cfg_.model_selection;
    report.model_name = modelLabel(cfg_.model_selection);

    double sum_det=0,sum_af=0,sum_fn=0,sum_bu=0,sum_gh=0,
           sum_emb=0,sum_match=0,sum_total=0;

    std::cout << "[Benchmark] " << report.model_name << " – "
              << num_frames << " frames\n";

    for (int i=0; i<num_frames; ++i) {
        cv::Mat frame;
        if (!cap.read(frame)||frame.empty()) break;
        FrameStats s;
        processFrame(frame, &s);
        ++report.total_frames;
        if (s.face_detected) ++report.frames_with_face;
        if (s.matched)       ++report.frames_matched;
        sum_det+=s.detection_ms; sum_af+=s.arcface_ms;
        sum_fn+=s.facenet_ms;    sum_bu+=s.buffalo_ms;
        sum_gh+=s.ghost_ms;      sum_emb+=s.total_embed_ms;
        sum_match+=s.match_ms;   sum_total+=s.total_ms;
        if (s.total_ms<report.min_total_ms) report.min_total_ms=s.total_ms;
        if (s.total_ms>report.max_total_ms) report.max_total_ms=s.total_ms;
        if ((i+1)%50==0) std::cout<<"  Frame "<<(i+1)<<"/"<<num_frames<<"\n";
    }
    cap.release();

    if (report.total_frames>0) {
        double n=report.total_frames;
        report.avg_detection_ms=sum_det/n;   report.avg_arcface_ms=sum_af/n;
        report.avg_facenet_ms=sum_fn/n;      report.avg_buffalo_ms=sum_bu/n;
        report.avg_ghost_ms=sum_gh/n;        report.avg_total_embed_ms=sum_emb/n;
        report.avg_match_ms=sum_match/n;     report.avg_total_ms=sum_total/n;
    }
    return report;
}

void Recognizer::runFullBenchmark(const Config& base_cfg,
                                   int camera_id, int num_frames,
                                   const std::string& output_csv)
{
    std::vector<ModelSelection> models = {
        ModelSelection::ARCFACE_ONLY, ModelSelection::FACENET_ONLY,
        ModelSelection::BOTH_MODELS,  ModelSelection::BUFFALO_L,
        ModelSelection::GHOSTFACENET, ModelSelection::ALL_MODELS
    };
    std::cout << "\n=== FULL BENCHMARK – "<<num_frames<<" frames each ===\n\n";
    for (auto m : models) {
        Config cfg = base_cfg;
        cfg.model_selection  = m;
        cfg.cooldown_seconds = 999999;
        cfg.top_k            = 1;
        Recognizer rec(cfg);
        auto report = rec.benchmark(camera_id, num_frames);
        report.print();
        report.saveCsv(output_csv);
    }
    std::cout << "\n[Benchmark] Saved: " << output_csv << "\n";
}

// ── Embeddings ────────────────────────────────────────────────────────────────

Recognizer::Embeddings Recognizer::generateEmbeddings(
    const cv::Mat& face_crop, FrameStats* stats)
{
    Embeddings out;
    if (arcface_engine_) {
        auto t=Clock::now();
        out.arcface=arcface_engine_->run(face_crop);
        if(stats) stats->arcface_ms=Ms(Clock::now()-t).count();
    }
    if (facenet_engine_) {
        auto t=Clock::now();
        out.facenet=facenet_engine_->run(face_crop);
        if(stats) stats->facenet_ms=Ms(Clock::now()-t).count();
    }
    if (buffalo_loaded_) {
        auto t=Clock::now();
        out.buffalo=runBuffalo(face_crop);
        if(stats) stats->buffalo_ms=Ms(Clock::now()-t).count();
    }
    if (ghost_loaded_) {
        auto t=Clock::now();
        out.ghost=runGhost(face_crop);
        if(stats) stats->ghost_ms=Ms(Clock::now()-t).count();
    }
    if (stats)
        stats->total_embed_ms=stats->arcface_ms+stats->facenet_ms+
                               stats->buffalo_ms+stats->ghost_ms;
    return out;
}

static cv::Mat prepareEmbeddingInput(const cv::Mat& face_crop, int size) {
    cv::Mat img;
    cv::resize(face_crop, img, cv::Size(size,size));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_32F, 1.0/255.0);
    cv::subtract(img, cv::Scalar(0.5f,0.5f,0.5f), img);
    cv::divide(img,   cv::Scalar(0.5f,0.5f,0.5f), img);
    std::vector<cv::Mat> ch(3); cv::split(img,ch);
    cv::Mat chw; cv::vconcat(ch,chw);
    return chw.reshape(1,{1,3,size,size});
}

cv::Mat Recognizer::runBuffalo(const cv::Mat& face_crop) {
    try {
        auto req=buffalo_compiled_.create_infer_request();
        cv::Mat inp=prepareEmbeddingInput(face_crop,112);
        ov::Tensor t(buffalo_compiled_.input().get_element_type(),{1,3,112,112},inp.data);
        req.set_input_tensor(t); req.infer();
        auto out=req.get_output_tensor(0);
        return cv::Mat(1,out.get_shape()[1],CV_32F,out.data<float>()).clone();
    } catch(const std::exception& e) {
        std::cerr<<"[Buffalo_L] "<<e.what()<<"\n"; return cv::Mat();
    }
}

cv::Mat Recognizer::runGhost(const cv::Mat& face_crop) {
    try {
        auto req=ghost_compiled_.create_infer_request();
        cv::Mat inp=prepareEmbeddingInput(face_crop,112);
        ov::Tensor t(ghost_compiled_.input().get_element_type(),{1,3,112,112},inp.data);
        req.set_input_tensor(t); req.infer();
        auto out=req.get_output_tensor(0);
        return cv::Mat(1,out.get_shape()[1],CV_32F,out.data<float>()).clone();
    } catch(const std::exception& e) {
        std::cerr<<"[GhostFaceNet] "<<e.what()<<"\n"; return cv::Mat();
    }
}

std::string Recognizer::modelLabel(ModelSelection m) {
    switch(m) {
        case ModelSelection::ARCFACE_ONLY:  return "ArcFace";
        case ModelSelection::FACENET_ONLY:  return "FaceNet512";
        case ModelSelection::BOTH_MODELS:   return "ArcFace+FaceNet";
        case ModelSelection::BUFFALO_L:     return "Buffalo_L";
        case ModelSelection::GHOSTFACENET:  return "GhostFaceNet";
        case ModelSelection::ALL_MODELS:    return "All_Models";
        default:                            return "Unknown";
    }
}

// ── Folder enrollment ─────────────────────────────────────────────────────────

std::string Recognizer::enrollFolder(const std::string& name,
                                      const std::string& folder_path,
                                      int* images_used_out)
{
    namespace fs = std::filesystem;

    if (!fs::exists(folder_path) || !fs::is_directory(folder_path))
        throw std::runtime_error("Folder not found: " + folder_path);

    // Supported image extensions
    static const std::vector<std::string> exts =
        {".jpg",".jpeg",".png",".bmp",".webp",".tiff",".tif"};

    // Collect all image paths
    std::vector<fs::path> image_paths;
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (std::find(exts.begin(), exts.end(), ext) != exts.end())
            image_paths.push_back(entry.path());
    }
    std::sort(image_paths.begin(), image_paths.end());

    if (image_paths.empty())
        throw std::runtime_error("No images found in: " + folder_path);

    std::cout << "[EnrollFolder] Found " << image_paths.size()
              << " images for '" << name << "'\n";

    // Accumulators for averaging
    cv::Mat sum_af, sum_fn, sum_bu, sum_gh;
    int count_af=0, count_fn=0, count_bu=0, count_gh=0;
    int images_used = 0;

    for (const auto& p : image_paths) {
        cv::Mat img = cv::imread(p.string());
        if (img.empty()) {
            std::cout << "  [skip] Cannot read: " << p.filename() << "\n";
            continue;
        }

        // Run face detection
        cv::Mat frame = img.clone();
        detector_->RunInference(frame);
        if (detector_->cropped_face.empty()) {
            std::cout << "  [skip] No face detected: " << p.filename() << "\n";
            continue;
        }

        auto embs = generateEmbeddings(detector_->cropped_face);

        // Accumulate — L2-normalise each embedding before summing
        auto accumulate = [](const cv::Mat& emb, cv::Mat& sum, int& cnt) {
            if (emb.empty()) return;
            cv::Mat e = emb.reshape(1, 1);
            e.convertTo(e, CV_32F);
            double n = cv::norm(e);
            if (n > 1e-8) e /= n;
            if (sum.empty()) sum = e.clone();
            else             sum += e;
            ++cnt;
        };

        accumulate(embs.arcface, sum_af, count_af);
        accumulate(embs.facenet, sum_fn, count_fn);
        accumulate(embs.buffalo, sum_bu, count_bu);
        accumulate(embs.ghost,   sum_gh, count_gh);

        ++images_used;
        std::cout << "  [ok] " << p.filename().string()
                  << "  (face detected)\n";
    }

    if (images_used == 0)
        throw std::runtime_error(
            "No valid face found in any image in: " + folder_path);

    // Average then L2-normalise the sum to get the final template
    auto finalise = [](cv::Mat& sum, int cnt) -> cv::Mat {
        if (sum.empty() || cnt == 0) return cv::Mat();
        cv::Mat avg = sum / static_cast<float>(cnt);
        double n = cv::norm(avg);
        if (n > 1e-8) avg /= n;
        return avg;
    };

    cv::Mat af = finalise(sum_af, count_af);
    cv::Mat fn = finalise(sum_fn, count_fn);
    cv::Mat bu = finalise(sum_bu, count_bu);
    cv::Mat gh = finalise(sum_gh, count_gh);

    std::string id = db_->enroll(name, af, fn, bu, gh);

    std::cout << "[EnrollFolder] Done — used " << images_used << "/"
              << image_paths.size() << " images"
              << "  ID: " << id << "\n";

    if (images_used_out) *images_used_out = images_used;
    return id;
}

// ── Database enrollment (all subfolders) ──────────────────────────────────────

int Recognizer::enrollDatabase(const std::string& faces_dir, bool re_enroll)
{
    namespace fs = std::filesystem;

    if (!fs::exists(faces_dir) || !fs::is_directory(faces_dir))
        throw std::runtime_error("faces_dir not found: " + faces_dir);

    // Collect existing names for duplicate detection
    std::vector<std::string> existing = db_->listNames();

    // Collect and sort subfolders for deterministic ordering
    std::vector<fs::path> subdirs;
    for (const auto& entry : fs::directory_iterator(faces_dir)) {
        if (entry.is_directory())
            subdirs.push_back(entry.path());
    }
    std::sort(subdirs.begin(), subdirs.end());

    if (subdirs.empty()) {
        std::cout << "[EnrollDB] No subfolders found in: " << faces_dir << "\n";
        return 0;
    }

    std::cout << "\n[EnrollDB] Found " << subdirs.size()
              << " person folder(s) in: " << faces_dir << "\n"
              << std::string(60, '-') << "\n";

    int enrolled = 0, skipped = 0, failed = 0;

    for (const auto& dir : subdirs) {
        // Folder name → person name (replace underscores with spaces)
        std::string raw  = dir.filename().string();
        std::string name = raw;
        std::replace(name.begin(), name.end(), '_', ' ');

        // Skip if already enrolled and re_enroll flag not set
        bool already = std::find(existing.begin(), existing.end(), name)
                       != existing.end();
        if (already && !re_enroll) {
            std::cout << "  [skip]    " << name
                      << " (already enrolled — use --re-enroll to update)\n";
            ++skipped;
            continue;
        }

        std::cout << "\n  ► Enrolling: " << name << "\n";
        try {
            int used = 0;
            std::string id = enrollFolder(name, dir.string(), &used);
            std::cout << "    ✔ ID=" << id
                      << "  images_used=" << used << "\n";
            ++enrolled;
        } catch (const std::exception& e) {
            std::cout << "    ✘ FAILED: " << e.what() << "\n";
            ++failed;
        }
    }

    std::cout << "\n" << std::string(60, '-') << "\n"
              << "[EnrollDB] Done."
              << "  enrolled=" << enrolled
              << "  skipped="  << skipped
              << "  failed="   << failed
              << "  total_db=" << db_->size() << "\n\n";

    return enrolled;
}
