#pragma once

#include <string>
#include <unordered_map>
#include <chrono>
#include <filesystem>
#include "face_database.hpp"

class AttendanceLogger {
public:
    explicit AttendanceLogger(
        const std::string& log_dir  = "attendance",
        int                cooldown = 60);
    ~AttendanceLogger() = default;

    bool log(const MatchResult& match, double inference_ms = 0.0);
    double secondsSinceLastLog(const std::string& person_id) const;
    int    todayCount() const;
    std::string todayLogPath() const;

private:
    std::string log_dir_;
    int         cooldown_seconds_;
    std::unordered_map<std::string,
        std::chrono::steady_clock::time_point> last_logged_;

    static std::string currentTimestamp();
    static std::string todayDateString();
    static std::string modelName(ModelSelection m);
    void ensureHeader(const std::string& path) const;
};
