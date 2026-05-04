#pragma once
// Extended ModelSelection that supersedes the one in face_embedding.hpp
// This header must be included BEFORE face_embedding.hpp in all attendance files.
// We achieve this by including it first in face_database.hpp which is included everywhere.

// To avoid conflict with the original enum in face_embedding.hpp,
// we wrap face_embedding.hpp inclusion with a macro that skips its enum.
// Since we cannot modify the original file, instead we use a different strategy:
// face_database.hpp defines ModelSelection FIRST, then face_embedding.hpp
// is included. However face_embedding.hpp uses include guards so we need
// to ensure our enum is defined before face_embedding.hpp's enum.
//
// Solution: create a forwarding wrapper that defines the full enum,
// then maps the original 3 values when calling FaceEmbeddingIntegration.
// The recognizer.cpp handles this mapping internally.

enum class ModelSelection {
    ARCFACE_ONLY,       // 1
    FACENET_ONLY,       // 2
    BOTH_MODELS,        // 3 - ArcFace + FaceNet512
    BUFFALO_L,          // 4
    GHOSTFACENET,       // 5
    ALL_MODELS          // 6 - All models averaged
};
