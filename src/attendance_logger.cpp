#include "attendance_logger.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <ctime>

AttendanceLogger::AttendanceLogger(const std::string& log_dir, int cooldown)
    : log_dir_(log_dir), cooldown_seconds_(cooldown)
{
    std::filesystem::create_directories(log_dir_);
    ensureHeader(todayLogPath());
}

bool AttendanceLogger::log(const MatchResult& match, double inference_ms) {
    if (!match.matched) return false;

    const std::string& pid = match.person_id;

    auto it = last_logged_.find(pid);
    if (it != last_logged_.end()) {
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - it->second).count();
        if (elapsed < cooldown_seconds_) return false;
    }

    std::string path = todayLogPath();
    ensureHeader(path);
    std::ofstream f(path, std::ios::app);
    if (!f) return false;

    f << currentTimestamp()               << ","
      << pid                              << ","
      << match.person_name               << ","
      << std::fixed << std::setprecision(4)
      << match.arcface_similarity        << ","
      << match.facenet_similarity        << ","
      << match.buffalo_similarity        << ","
      << match.ghost_similarity          << ","
      << match.combined_similarity       << ","
      << modelName(match.model_used)     << ","
      << std::setprecision(1)
      << inference_ms                    << "\n";

    last_logged_[pid] = std::chrono::steady_clock::now();

    std::cout << "[Logger] " << match.person_name
              << "  model=" << modelName(match.model_used)
              << "  sim="   << std::setprecision(3) << match.combined_similarity
              << "  "       << std::setprecision(1) << inference_ms << "ms\n";
    return true;
}

double AttendanceLogger::secondsSinceLastLog(const std::string& person_id) const {
    auto it = last_logged_.find(person_id);
    if (it == last_logged_.end()) return -1.0;
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - it->second).count();
}

int AttendanceLogger::todayCount() const {
    std::ifstream f(todayLogPath());
    if (!f) return 0;
    int count = -1;
    std::string line;
    while (std::getline(f, line)) ++count;
    return std::max(0, count);
}

std::string AttendanceLogger::todayLogPath() const {
    return log_dir_ + "/attendance_" + todayDateString() + ".csv";
}

std::string AttendanceLogger::currentTimestamp() {
    auto now  = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

std::string AttendanceLogger::todayDateString() {
    auto now  = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time), "%Y-%m-%d");
    return oss.str();
}

std::string AttendanceLogger::modelName(ModelSelection m) {
    switch (m) {
        case ModelSelection::ARCFACE_ONLY:  return "ArcFace";
        case ModelSelection::FACENET_ONLY:  return "FaceNet512";
        case ModelSelection::BOTH_MODELS:   return "ArcFace+FaceNet";
        case ModelSelection::BUFFALO_L:     return "Buffalo_L";
        case ModelSelection::GHOSTFACENET:  return "GhostFaceNet";
        case ModelSelection::ALL_MODELS:    return "All_Models";
        default:                            return "Unknown";
    }
}

void AttendanceLogger::ensureHeader(const std::string& path) const {
    if (std::filesystem::exists(path)) return;
    std::ofstream f(path);
    f << "timestamp,person_id,person_name,"
         "arcface_sim,facenet_sim,buffalo_sim,ghost_sim,"
         "combined_sim,model_used,inference_ms\n";
}
