#include "recognizer.hpp"
#include <iostream>
#include <string>
#include <filesystem>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

static void printUsage(const char* prog) {
    std::cout <<
        "\nUsage:\n"
        "  " << prog << " enroll        --name <name> --image <path>  [options]\n"
        "  " << prog << " enroll-folder --name <name> --folder <dir>  [options]\n"
        "  " << prog << " enroll-db     --faces-dir <dir>             [options]\n"
        "  " << prog << " recognize     --camera <id>                 [options]\n"
        "  " << prog << " benchmark     --camera <id> --frames <N>    [options]\n"
        "\nenroll-db (recommended):\n"
        "  Enrolls every person subfolder inside <dir> in one shot.\n"
        "  Folder name becomes the person's name (underscores → spaces).\n"
        "  Example structure:\n"
        "    faces/\n"
        "      Ahmed_Gamal/   <- 10-30 images of Ahmed\n"
        "      Sara_Mohamed/  <- 10-30 images of Sara\n"
        "  Each person's embeddings are averaged across all their images.\n"
        "  Add --re-enroll to overwrite existing entries.\n"
        "\nenroll-folder:\n"
        "  Enroll a single person from a folder of images.\n"
        "  Pass --name explicitly.\n"
        "\nModel options (--model):\n"
        "  1 = ArcFace only\n"
        "  2 = FaceNet512 only\n"
        "  3 = ArcFace + FaceNet512\n"
        "  4 = Buffalo_L\n"
        "  5 = GhostFaceNet\n"
        "  6 = All models combined\n"
        "  0 = [benchmark only] run all models back-to-back\n"
        "\nOther options:\n"
        "  --thresh     <float>  Cosine similarity threshold  (default: 0.35)\n"
        "  --cooldown   <secs>   Attendance cooldown seconds  (default: 60)\n"
        "  --device     <str>    OpenVINO device AUTO/CPU/GPU (default: AUTO)\n"
        "  --db         <dir>    Face database directory      (default: db)\n"
        "  --faces-dir  <dir>    Root folder with per-person subfolders\n"
        "  --re-enroll          Overwrite existing DB entries\n"
        "  --key        <path>   Encryption key path\n"
        "  --yolo       <path>   YOLO model .xml path\n"
        "  --arcface    <path>   ArcFace model .xml path\n"
        "  --facenet    <path>   FaceNet512 model .xml path\n"
        "  --buffalo    <path>   Buffalo_L model .xml path\n"
        "  --ghost      <path>   GhostFaceNet model .xml path\n"
        "  --frames     <N>      Frames for benchmark         (default: 300)\n"
        "  --topk       <K>      Top-K candidates to return   (default: 5)\n\n";
}

struct Args {
    std::string mode;
    std::string name, image_path, folder_path, faces_dir;
    int   camera_id        = 0;
    int   model_sel        = 1;
    float thresh           = 0.35f;
    int   cooldown         = 60;
    bool  re_enroll        = false;
    std::string device     = "AUTO";
    std::string db_dir     = "db";
    std::string key_path, yolo_path, arcface_path, facenet_path,
                buffalo_path, ghost_path;
    int   benchmark_frames = 300;
    int   top_k            = 5;
};

static Args parseArgs(int argc, char** argv) {
    Args a;
    if (argc < 2) return a;
    a.mode = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string k = argv[i];
        std::string v = (i+1 < argc) ? argv[i+1] : "";
        if      (k=="--name")       { a.name            = v; ++i; }
        else if (k=="--image")      { a.image_path       = v; ++i; }
        else if (k=="--folder")     { a.folder_path      = v; ++i; }
        else if (k=="--faces-dir")  { a.faces_dir        = v; ++i; }
        else if (k=="--re-enroll")  { a.re_enroll        = true;   }
        else if (k=="--camera")     { a.camera_id        = std::stoi(v); ++i; }
        else if (k=="--model")      { a.model_sel        = std::stoi(v); ++i; }
        else if (k=="--thresh")     { a.thresh           = std::stof(v); ++i; }
        else if (k=="--cooldown")   { a.cooldown         = std::stoi(v); ++i; }
        else if (k=="--device")     { a.device           = v; ++i; }
        else if (k=="--db")         { a.db_dir           = v; ++i; }
        else if (k=="--key")        { a.key_path         = v; ++i; }
        else if (k=="--yolo")       { a.yolo_path        = v; ++i; }
        else if (k=="--arcface")    { a.arcface_path     = v; ++i; }
        else if (k=="--facenet")    { a.facenet_path     = v; ++i; }
        else if (k=="--buffalo")    { a.buffalo_path     = v; ++i; }
        else if (k=="--ghost")      { a.ghost_path       = v; ++i; }
        else if (k=="--frames")     { a.benchmark_frames = std::stoi(v); ++i; }
        else if (k=="--topk")       { a.top_k            = std::stoi(v); ++i; }
    }
    return a;
}

// Resolve checkpoints dir relative to the executable location.
// When run from the project root as ./build/attendance_system,
// this resolves to <project_root>/checkpoints/
#include <climits>
#include <unistd.h>

static std::string getProjectRoot() {
    // Walk up from the executable to find the project root
    // (the directory that contains CMakeLists.txt and checkpoints/)
    char exe[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exe, sizeof(exe)-1);
    if (len < 0) return ".";
    exe[len] = '\0';
    std::filesystem::path p(exe);
    // executable is at <root>/build/attendance_system — go up twice
    return (p.parent_path().parent_path()).string();
}

static const std::string CKPT = getProjectRoot() + "/checkpoints";
static const std::string KEY_PATH = getProjectRoot() + "/encryption.key";

static Recognizer::Config buildConfig(const Args& a) {
    Recognizer::Config cfg;
    cfg.yolo_model_path    = a.yolo_path.empty()
        ? CKPT + "/yolov8nface_openvino/yolov8nface.xml" : a.yolo_path;
    cfg.arcface_model_path = a.arcface_path.empty()
        ? CKPT + "/arcface_openvino/arcface_openvino.xml" : a.arcface_path;
    cfg.facenet_model_path = a.facenet_path.empty()
        ? CKPT + "/facenet_openvino/facenet512.xml" : a.facenet_path;
    cfg.buffalo_model_path = a.buffalo_path.empty()
        ? CKPT + "/buffalo_openvino/buffalo_l.xml" : a.buffalo_path;
    cfg.ghost_model_path   = a.ghost_path.empty()
        ? CKPT + "/ghost_openvino/ghostfacenet.xml" : a.ghost_path;
    cfg.key_path           = a.key_path.empty()
        ? KEY_PATH
        : a.key_path;
    cfg.device             = a.device;
    cfg.db_dir             = a.db_dir;
    cfg.recognition_thresh = a.thresh;
    cfg.top_k              = a.top_k;
    cfg.cooldown_seconds   = a.cooldown;

    switch (a.model_sel) {
        case 1:  cfg.model_selection = ModelSelection::ARCFACE_ONLY;  break;
        case 2:  cfg.model_selection = ModelSelection::FACENET_ONLY;  break;
        case 3:  cfg.model_selection = ModelSelection::BOTH_MODELS;   break;
        case 4:  cfg.model_selection = ModelSelection::BUFFALO_L;     break;
        case 5:  cfg.model_selection = ModelSelection::GHOSTFACENET;  break;
        case 6:  cfg.model_selection = ModelSelection::ALL_MODELS;    break;
        default: cfg.model_selection = ModelSelection::ARCFACE_ONLY;
    }
    return cfg;
}

// ── Single-image enroll ───────────────────────────────────────────────────────

static int runEnroll(const Args& a) {
    if (a.name.empty()) {
        std::cerr << "[enroll] --name required\n"; return 1; }
    if (!std::filesystem::exists(a.image_path)) {
        std::cerr << "[enroll] image not found: " << a.image_path << "\n"; return 1; }
    cv::Mat image = cv::imread(a.image_path);
    if (image.empty()) {
        std::cerr << "[enroll] failed to load image\n"; return 1; }
    Args ea = a; ea.model_sel = 6;
    Recognizer rec(buildConfig(ea));
    std::string id = rec.enroll(a.name, image);
    std::cout << "\n✔  Enrolled '" << a.name << "'  ID: " << id << "\n";
    return 0;
}

// ── Single-folder enroll ──────────────────────────────────────────────────────

static int runEnrollFolder(const Args& a) {
    if (a.name.empty()) {
        std::cerr << "[enroll-folder] --name required\n"; return 1; }
    if (a.folder_path.empty()) {
        std::cerr << "[enroll-folder] --folder required\n"; return 1; }
    if (!std::filesystem::is_directory(a.folder_path)) {
        std::cerr << "[enroll-folder] not a directory: " << a.folder_path << "\n";
        return 1; }
    Args ea = a; ea.model_sel = 6;
    Recognizer rec(buildConfig(ea));
    int used = 0;
    std::string id = rec.enrollFolder(a.name, a.folder_path, &used);
    std::cout << "\n✔  Enrolled '" << a.name << "'"
              << "  images_used=" << used
              << "  ID: " << id << "\n";
    return 0;
}

// ── Full database enroll ──────────────────────────────────────────────────────

static int runEnrollDatabase(const Args& a) {
    if (a.faces_dir.empty()) {
        std::cerr << "[enroll-db] --faces-dir required\n";
        std::cerr << "  Example: --faces-dir ~/Desktop/faces\n";
        std::cerr << "  Each subfolder name becomes the person's name.\n";
        return 1;
    }
    if (!std::filesystem::is_directory(a.faces_dir)) {
        std::cerr << "[enroll-db] not a directory: " << a.faces_dir << "\n";
        return 1;
    }
    Args ea = a; ea.model_sel = 6;
    Recognizer rec(buildConfig(ea));
    int n = rec.enrollDatabase(a.faces_dir, a.re_enroll);
    std::cout << "\n✔  enrollDatabase complete — " << n << " new person(s) enrolled.\n";
    return 0;
}

// ── Recognize ─────────────────────────────────────────────────────────────────

static int runRecognize(const Args& a) {
    Recognizer rec(buildConfig(a));
    if (rec.database().empty())
        std::cout << "[WARNING] Database empty – enroll people first.\n";

    cv::VideoCapture cap(a.camera_id);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera " << a.camera_id << "\n"; return 1; }
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    std::cout << "\n[recognize] Press 'q' to quit.\n"
              << "Attendance log: " << rec.logger().todayLogPath() << "\n\n";
    cv::namedWindow("Attendance System", cv::WINDOW_NORMAL);

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) break;
        FrameStats stats;
        MatchResult result = rec.processFrame(frame, &stats);
        if (stats.face_detected) {
            if (result.matched)
                std::cout << "\r✔ " << result.person_name
                          << "  sim=" << std::fixed << std::setprecision(3)
                          << result.combined_similarity
                          << "  " << std::setprecision(1)
                          << stats.total_ms << "ms   " << std::flush;
            else
                std::cout << "\r✘ Unknown  " << std::flush;
        }
        cv::imshow("Attendance System", frame);
        if (cv::waitKey(1) == 'q') break;
    }
    cap.release();
    cv::destroyAllWindows();
    std::cout << "\n[Done] Log: " << rec.logger().todayLogPath()
              << " (" << rec.logger().todayCount() << " records)\n";
    return 0;
}

// ── Benchmark ─────────────────────────────────────────────────────────────────

static int runBenchmark(const Args& a) {
    std::string csv = "benchmark_results.csv";
    if (a.model_sel == 0) {
        Recognizer::Config cfg = buildConfig(a);
        Recognizer::runFullBenchmark(cfg, a.camera_id, a.benchmark_frames, csv);
    } else {
        Recognizer::Config cfg = buildConfig(a);
        cfg.cooldown_seconds = 999999;
        Recognizer rec(cfg);
        auto report = rec.benchmark(a.camera_id, a.benchmark_frames);
        report.print();
        report.saveCsv(csv);
    }
    std::cout << "Benchmark saved: " << csv << "\n";
    return 0;
}

// ── Entry point ───────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc < 2) { printUsage(argv[0]); return 1; }
    Args a = parseArgs(argc, argv);
    try {
        if      (a.mode == "enroll")        return runEnroll(a);
        else if (a.mode == "enroll-folder") return runEnrollFolder(a);
        else if (a.mode == "enroll-db")     return runEnrollDatabase(a);
        else if (a.mode == "recognize")     return runRecognize(a);
        else if (a.mode == "benchmark")     return runBenchmark(a);
        else {
            std::cerr << "Unknown mode: " << a.mode << "\n";
            printUsage(argv[0]);
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n"; return 1;
    }
}
