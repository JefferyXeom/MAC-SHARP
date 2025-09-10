//
// MacSharp.hpp
// Public declarations for registration after extracting monitoring & evaluation.
//

#pragma once

#include "MacData.hpp"
#include "MacConfig.hpp"

class MacMonitor; // forward declare

/**
 * @brief Main registration pipeline entry.
 * @details
 *   MAC-SHARP flow:
 *     load data -> build graph -> compute graph weights -> find maximal cliques ->
 *     generate/filter/cluster hypotheses -> select & (optional) refine -> final evaluation.
 *
 * @param macConfig Global configuration (execution/algorithm/evaluation).
 * @param macData   Data container (clouds, correspondences, GT).
 * @param macResult Result container (RE/TE/F1/timing/...).
 * @param monitor   Run-time monitor for CSV timing rows; logs are handled externally.
 * @return true if registration is successful (by evaluation criteria), false otherwise.
 */
bool registration(const MacConfig& macConfig, MacData& macData, MacResult& macResult,
                  MacMonitor& monitor);
