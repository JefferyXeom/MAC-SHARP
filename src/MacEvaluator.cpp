//
// Created by Jeffery_Xeom on 2025/9/7.
// Project: MAC_SHARP
// File: MacEvaluator.cpp
//

#include <algorithm>  // for std::min
#include <pcl/common/transforms.h>

#include "../include/MacEvaluator.hpp"
#include "../include/MacUtils.hpp"

bool MacEvaluator::validateFinalTransform(const Eigen::Matrix4f& finalTransform,
                                          const MacData& data,
                                          const MacConfig& config,
                                          MacResult& result) {
    LOG_INFO("--- Final Validation ---");

    const DatasetConfig& ds = config.getCurrentDatasetConfig();
    const float rotThreshDeg  = ds.reThresh;      // degrees
    const float transThreshM  = ds.teThresh;      // meters
    const float inlierThreshM = ds.getActualInlierThreshold(); // meters

    // evaluationEst requires non-const refs for est/gt
    Eigen::Matrix4f estCopy = finalTransform;
    Eigen::Matrix4f gtCopy  = data.gtTransform;
    float RE = 0.0f, TE = 0.0f;
    const bool pass = evaluationEst(estCopy, gtCopy, rotThreshDeg, transThreshM, RE, TE);
    result.RE = RE;
    result.TE = TE;

    LOG_INFO("Final Rotation Error (RE): " << RE << " deg  (Threshold: " << rotThreshDeg << " deg)");
    LOG_INFO("Final Translation Error (TE): " << TE << " m   (Threshold: " << transThreshM  << " m)");

    // Inlier stats
    const pcl::PointCloud<pcl::PointXYZ>::Ptr srcTrans(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::transformPointCloud(*(data.cloudSrc), *srcTrans, finalTransform);

    // int predicted = 0, correct = 0;
    // const size_t N = data.totalCorresNum;
    // for (size_t i = 0; i < N; ++i) {
    //     // Distance in meters after applying final transform to source point i.
    //     if (const float dist = getDistance(srcTrans->points[i], data.cloudTgt->points[i]); dist < inlierThreshM) {
    //         ++predicted;
    //         // Guard label array bound as well.
    //         if (!data.gtInlierLabels.empty() && i < data.gtInlierLabels.size() && data.gtInlierLabels[i] == 1) ++correct;
    //     }
    // }
    //
    // const float precision = (predicted > 0) ? static_cast<float>(correct) / static_cast<float>(predicted) : 0.0f;
    // const float recall    = (data.gtInlierCount > 0) ? static_cast<float>(correct) / static_cast<float>(data.gtInlierCount) : 0.0f;
    // const float f1        = (precision + recall > 0.0f) ? 2.f * (precision * recall) / (precision + recall) : 0.0f;
    //
    // LOG_INFO("Final Inliers -> Predicted: " << predicted
    //          << ", Correct: " << correct
    //          << ", GT Total: " << data.gtInlierCount);
    // LOG_INFO("Thresholds (deg/m): RE_th=" << rotThreshDeg
    //      << ", TE_th=" << transThreshM
    //      << ", Inlier_th=" << inlierThreshM);
    // LOG_INFO("Precision: " << precision * 100.0f << "%, Recall: " << recall * 100.0f << "%, F1: " << f1);

    // result.predictedInlier = {precision, recall, f1};
    // [Added] Persist pass flag and the thresholds for later reporting.
    // NOTE: These fields may need to be added to MacResult (we will patch MacResult later if missing).
    result.evalPass = pass;                   // bool: overall pass/fail by thresholds
    result.reThreshDegUsed = rotThreshDeg;    // float: RE threshold used (degrees)
    result.teThreshMUsed   = transThreshM;    // float: TE threshold used (meters)
    result.inlierThreshMUsed = inlierThreshM; // float: inlier distance threshold used (meters)
    // [Added] Persist raw counts so the caller can log/aggregate as needed.
    // result.inlierPredictedCount = predicted;          // number of predicted inliers
    // result.inlierCorrectCount   = correct;            // predicted inliers that match GT inlier label
    result.inlierGtTotalCount   = data.gtInlierCount; // total GT inliers

    // [Note] You currently use 'predicatedInlier' which likely intended to be 'predictedInlier'.
    // We'll keep the existing name here to avoid breaking builds, but I recommend renaming the field
    // to 'predictedInlier' in CommonTypes.hpp later (I can provide a follow-up patch).


    if (pass) {
        LOG_INFO(GREEN << "Registration validation successful (RE/TE within thresholds)." << RESET);
    } else {
        LOG_WARNING(YELLOW << "Registration validation failed (RE/TE exceeds thresholds)." << RESET);
    }
    return pass;
}
