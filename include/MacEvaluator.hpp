//
// Created by Jeffery_Xeom on 2025/9/7.
// Project: MAC_SHARP
// File: MacEvaluator.hpp
//
//
// MacEvaluator.hpp
// Evaluation-only helper: final RE/TE and inlier stats under dataset thresholds.
//

#pragma once

#include <Eigen/Dense>
#include "MacData.hpp"
#include "MacConfig.hpp"

class MacEvaluator {
public:
    /**
     * @brief Evaluate the final transform (optionally ICP-refined) against GT.
     * @details
     *   - Computes RE (deg) and TE (m) via evaluationEst().
     *   - Counts inliers under dataset inlier threshold (m).
     *   - Stores Precision/Recall/F1 into result.predicatedInlier.
     *   - Does NOT affect the algorithmic path (evaluation-only).
     *
     * @return true if RE<=rotThresh and TE<=transThresh, false otherwise.
     */
    static bool validateFinalTransform(const Eigen::Matrix4f& finalTransform,
                                       const MacData& data,
                                       const MacConfig& config,
                                       MacResult& result);
};
