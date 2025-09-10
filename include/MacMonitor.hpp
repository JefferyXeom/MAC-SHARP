//
// Created by Jeffery_Xeom on 2025/9/7.
// Project: MAC_SHARP
// File: MacRunMonitor.hpp
//
// MacMonitor.hpp
// Unified monitor: hierarchical auto-timing + CSV + tee-logging + memory sampling.
// C++17. Cross-platform (Windows / Linux).
//
// Usage (inclusive parent timing; exclusive is derived):
//   MacMonitor mon;
//   mon.setProgramStart(std::chrono::high_resolution_clock::now());
//   mon.init(macConfig);
//
//   mon.enter("constructGraph");           // L1 starts (inclusive)
//     mon.enter("firstOrder");             // L2 starts (inclusive, parent keeps running)
//       ... algorithm ...
//     mon.record();                        // write row for "firstOrder" (L2), pop L2
//     mon.enter("secondOrder");
//       ... algorithm ...
//     mon.record();                        // write row for "secondOrder" (L2), pop L2
//   mon.record();                          // write row for "constructGraph" (L1), pop L1
//
//   // early exit example:
//   mon.enter("generateRtHypotheses");
//   if (failed) { mon.cancel(); return false; } // pop without CSV
//   ... work ...
//   mon.record();
//
//   // whole-session meta row (optional):
//   mon.recordWholeSession(totalSecondsFromYourOuterTimer);
//
//   mon.shutdown();
//
// CSV columns:
//   level,path,stage,currentS,inclusiveS,exclusiveS,totalS_wall,totalS_stages,rssMb,peakRssMb
// Semantics:
//   - level:        current depth (1-based)
//   - path:         joined "a/b/c"
//   - stage:        leaf name (c)
//   - currentS:     equals inclusiveS (kept for backward naming compatibility)
//   - inclusiveS:   time from enter() to record(), inclusive of children
//   - exclusiveS:   inclusiveS minus sum(inclusiveS of direct children)
//   - totalS_wall:  wall-clock seconds since programStart
//   - totalS_stages:cumulative sum of recorded inclusiveS (pure algorithm curve)
//   - rssMb:        current resident set size (MB)
//   - peakRssMb:    peak resident set size (MB)
//
// Design notes:
//   * Inclusive timing for the parent means the parent clock never pauses while children run.
//   * We track per-frame 'childrenInclusiveSum'; when a child is recorded, its inclusive time
//     is added to the parent’s accumulator. Then exclusive = inclusive - childrenInclusiveSum.
//   * All IO/memory sampling happen AFTER we stop the stage timer, so they do not pollute timing.
//   * This class is not thread-safe. Use from the main sequential flow.
//

#pragma once

// [Added] Required headers for types used below
#include <string>
#include <vector>
#include <utility>
#include <fstream>
#include <chrono>
// #include <sstream>
// #include <streambuf>
#include <iomanip>

#include "MacConfig.hpp"  // for MonitorGranularity / MonitorMetrics / MonitorTreeCharset / MacConfig

/**
* @brief The MacMonitor provides hierarchical timing (inclusive/exclusive),
* memory sampling, CSV logging with a "note" JSON column (for metrics),
* tee logging to file, and a time-tree console summary.
*/
class MacMonitor {
public:
    MacMonitor();
    ~MacMonitor();
    // -------------------- Current Monitor (global access point) --------------------
    /**
     * @brief Set the process-wide current monitor pointer.
     *        This enables modules (MacData/MacGraph/MacRtHypothesis) to use monitor without passing it around.
     * @note  Not thread-safe by design. Call this at the beginning of a run and reset to nullptr at the end.
     */
    static void SetCurrent(MacMonitor* m) noexcept;

    /**
     * @brief Get the process-wide current monitor pointer.
     * @return The pointer passed via SetCurrent(), or nullptr if unset.
     */
    static MacMonitor* Current() noexcept;

    // Initialize sinks (tee log file + CSV). Output dir must exist.
    // Uses from MacConfig:
    //   - currentDatasetName (datasetToRun)
    //   - cloudSrcPath / cloudTgtPath (for base names)
    //   - outputPath (dir to write files)
    //   - flagNoLogs (disable log file when true)
    void init(const MacConfig& cfg);
    // Close CSV and restore std::cout/err. Warn & drop any unpaired stages.
    void shutdown();
    // Establish baseline for totalS_wall (wall-clock axis).
    void setProgramStart(std::chrono::high_resolution_clock::time_point ts) noexcept;

    // -------- Hierarchical auto-timing API (enter → record(pop) / cancel(pop)) --------
    // Begin a stage: push a frame and start its inclusive timer.
    void enter(const std::string& stageName);
    // Finish the current stage: stop its timer, write a CSV row, pop.
    // Also:
    //   - totalS_stages += inclusiveS
    //   - add inclusiveS to parent.childrenInclusiveSum (if parent exists)
    void record();
    // Early-exit: pop without writing a CSV row.
    void cancel();
    // Pop all remaining frames without writing rows (used in shutdown safety).
    void cancelAll();

    // Optional: write a synthetic "wholeSession" row (level=1).
    // 'totalSeconds' should be measured by your outer full-run timer if you have one.
    void recordWholeSession(double totalSeconds);

    // ===================== Metrics Note API (public) =====================
    // These do NOT immediately write to CSV; the decision to persist
    // is taken in record() based on monitor.metrics and evaluation.enabled.
    void setNote(const std::string& note);            // replace free-form note for current frame
    void clearNote();                                  // clear both note and kv pairs for current frame
    void setKV(const std::string& key, const std::string& value); // append/replace string metric
    void setKV(const std::string& key, double value);  // numeric overload (kept unquoted in JSON)
    void setKV(const std::string& key, int value);     // numeric overload
    void setKV(const std::string& key, bool value);    // boolean overload

    // ===================== Time-Tree Summary (public) =====================
    // Print an indented hierarchical summary to console (tee picks it to file).
    void dumpTimeTree() const;

    // Dump all currently open stages for debugging (no-op if stack is empty).
    void debugDumpOpenStages() const noexcept;

    // Accessors
    [[nodiscard]] int currentDepth() const noexcept { return static_cast<int>(stack_.size()); }
    [[nodiscard]] bool hasOpenStage() const noexcept { return !stack_.empty(); }
    [[nodiscard]] const std::string& logPath() const noexcept { return logPath_; }
    [[nodiscard]] const std::string& csvPath() const noexcept { return csvPath_; }

private:
    // Per-frame bookkeeping for inclusive/exclusive calculation.
    struct Frame {
        std::string name;                                       // stage name
        std::chrono::high_resolution_clock::time_point start;   // start timestamp
        double childrenInclusiveSum{0.0};                       // sum of inclusive seconds of direct children

        // [Added] New fields to support granularity filtering and metrics
        int  level = 0;              // 1-based depth when the frame is pushed
        bool recordable = true;      // whether this frame is recorded to CSV/time-tree (CORE vs ALL)
        std::string noteText;        // free-form note (merged into CSV "note" as "_note")
        std::vector<std::pair<std::string,std::string>> kvPairs; // key-value metrics for CSV "note"
    };

    // ---------------- Tee logger (console + file) ----------------
    // Tee streambuf
    // For teeing stdout/stderr to both console and file without changing your LOG_* macros.
    struct TeeBuf final : public std::streambuf {
        TeeBuf(std::streambuf* sb1, std::streambuf* sb2) : sb1_(sb1), sb2_(sb2) {}
    protected:
        int overflow(const int c) override {
            if (c == EOF) return !EOF;
            const int r1 = sb1_ ? sb1_->sputc(c) : c;
            const int r2 = sb2_ ? sb2_->sputc(c) : c;
            return (r1 == EOF || r2 == EOF) ? EOF : c;
        }
        int sync() override {
            const int r1 = sb1_ ? sb1_->pubsync() : 0;
            const int r2 = sb2_ ? sb2_->pubsync() : 0;
            return (r1 == 0 && r2 == 0) ? 0 : -1;
        }
    private:
        std::streambuf* sb1_;
        std::streambuf* sb2_;
    };

    struct TeeLogger {
        TeeLogger() = default;
        ~TeeLogger() { disable(); }
        void enable(const std::string& filePath, bool noFile);
        void disable();
    private:
        std::ofstream file_;
        std::streambuf* coutOrig_{nullptr};
        std::streambuf* cerrOrig_{nullptr};
        TeeBuf* teeCout_{nullptr};
        TeeBuf* teeCerr_{nullptr};
    };

    // [Added] Recorded node for time-tree print out.
    struct Recorded {
        int level = 0;
        std::string name;
        double inclusiveS = 0.0;
        double exclusiveS = 0.0;
        double rssMB = -1.0;
        double peakMB = -1.0;
        std::string noteJson; // compact JSON produced from kvPairs + noteText
    };

    // Global current monitor (process-wide). Kept simple; no atomic/locks for now.
    static MacMonitor* s_current;

    // Helpers
    void buildFileNames(const MacConfig& cfg);
    void openCsv();
    void writeCsvHeader();
    void appendCsvRow(int level,
                      const std::string& pathJoined,
                      const std::string& leafStage,
                      double inclusiveS,
                      double exclusiveS,
                      double totalWallS,
                      double totalStagesS,
                      double rssMb,
                      double peakRssMb,
                      const std::string& note);

    // Memory sampler implemented in .cpp (cross-platform).
    static std::pair<double,double> sampleMemoryMb(); // {rssMb, peakRssMb}
    // Path helpers (you already have them implemented in .cpp)
    static std::string baseName(const std::string& path);
    static std::string joinPath(const std::vector<std::string>& parts, char sep = '/');

    // ===================== Private static helpers =====================
    // Implemented in .cpp as MacMonitor::jsonEscape/buildNoteJson/treePrefix
    static std::string jsonEscape(const std::string& s);
    static std::string buildNoteJson(const std::vector<std::pair<std::string,std::string>>& kvs,
                                     const std::string& extraNote);
    static std::string treePrefix(int depth,bool isLast,bool useUnicode);


    // Wall-clock baseline
    std::chrono::high_resolution_clock::time_point programStart_{};
    bool programStartSet_{false};

    // ===================== Runtime containers =====================
    // Hierarchical stack (inclusive timing; exclusive derived)
    std::vector<Frame> stack_;          // timing stack
    std::vector<Recorded> recorded_;    // flat list for time-tree (in record order)

    // Sum of all recorded inclusive seconds (pure algorithm curve)
    double totalStagesSeconds_{0.0};

    // Output sinks
    std::string logPath_;
    std::string csvPath_;
    std::ofstream csv_;
    TeeLogger tee_;

    // Naming cache from config
    std::string datasetName_;
    std::string srcBase_;
    std::string tgtBase_;
    std::string outDir_;
    bool noLogFile_{false};

    // ===================== Cached runtime strategy =====================
    bool monitorEnabled_     = true;                     // master switch (monitor.enabled)
    MonitorGranularity granularity_ = MonitorGranularity::CORE; // CORE / ALL
    int  coreDepth_          = 2;                        // used only when CORE
    MonitorMetrics metricsPolicy_ = MonitorMetrics::AUTO;        // OFF / AUTO / ALL

    bool timeTreeEnabled_    = true;
    bool timeTreeUnicode_    = true;                     // charset=UNICODE => true; ASCII => false
    bool timeTreeShowMemory_ = true;
    bool timeTreeShowNotes_  = true;
    int  timeTreeMaxDepth_   = -1;

    bool evalEnabled_        = true; // evaluation.enabled (for metrics=AUTO gating)
};

// ========================= Convenience inline wrappers (no-ops if Current()==nullptr) =========================
// These helpers let modules call monitor without passing a reference. They simply forward to MacMonitor::Current().

inline void MON_ENTER(const std::string& stageName) {
    if (MacMonitor* m = MacMonitor::Current()) m->enter(stageName);
}
inline void MON_RECORD() {
    if (MacMonitor* m = MacMonitor::Current()) m->record();
}
inline void MON_CANCEL() {
    if (MacMonitor* m = MacMonitor::Current()) m->cancel();
}
inline void MON_SET_NOTE(const std::string& note) {
    if (MacMonitor* m = MacMonitor::Current()) m->setNote(note);
}
inline void MON_CLEAR_NOTE() {
    if (MacMonitor* m = MacMonitor::Current()) m->clearNote();
}
inline void MON_SET_KV(const std::string& key, const std::string& value) {
    if (MacMonitor* m = MacMonitor::Current()) m->setKV(key, value);
}
inline void MON_SET_KV(const std::string& key, double value) {
    if (MacMonitor* m = MacMonitor::Current()) m->setKV(key, value);
}
inline void MON_SET_KV(const std::string& key, int value) {
    if (MacMonitor* m = MacMonitor::Current()) m->setKV(key, value);
}
inline void MON_SET_KV(const std::string& key, bool value) {
    if (MacMonitor* m = MacMonitor::Current()) m->setKV(key, value);
}
// ==============================================================================================================
