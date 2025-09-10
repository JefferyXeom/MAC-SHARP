//
// Created by Jeffery_Xeom on 2025/9/7.
// Project: MAC_SHARP
// File: MacRunMonitor.cpp
////
// MacRunMonitor.cpp
//
#include "MacMonitor.hpp"

#include <iostream>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <cctype>   // for std::isdigit


#if defined(_WIN32)
  #define NOMINMAX
  #include <windows.h>
  #include <psapi.h>
  #pragma comment(lib, "psapi.lib")
#elif defined(__linux__)
  #include <sys/resource.h>
  #include <unistd.h>
  #include <fstream>
#endif
// [Added] Extra headers for formatting, mutex and platform checks
#include <sstream>
#include <iomanip>
#include <mutex>

// Define the process-wide current monitor pointer.
// Keep it simple (no atomics/locks). Set at run start; reset at run end.
MacMonitor* MacMonitor::s_current = nullptr;
void MacMonitor::SetCurrent(MacMonitor* m) noexcept {
    s_current = m;
}

MacMonitor* MacMonitor::Current() noexcept {
    return s_current;
}

void MacMonitor::TeeLogger::enable(const std::string& filePath, bool noFile) {
    if (noFile) return; // respect cfg.flagNoLogs
    file_.open(filePath, std::ios::out);
    if (!file_.is_open()) {
        std::cerr << "[WARN] Cannot open log file: " << filePath << "\n";
        return;
    }
    coutOrig_ = std::cout.rdbuf();
    cerrOrig_ = std::cerr.rdbuf();
    teeCout_ = new TeeBuf(coutOrig_, file_.rdbuf());
    teeCerr_ = new TeeBuf(cerrOrig_, file_.rdbuf());
    std::cout.rdbuf(teeCout_);
    std::cerr.rdbuf(teeCerr_);
}

void MacMonitor::TeeLogger::disable() {
    if (coutOrig_) std::cout.rdbuf(coutOrig_);
    if (cerrOrig_) std::cerr.rdbuf(cerrOrig_);
    coutOrig_ = nullptr; cerrOrig_ = nullptr;
    delete teeCout_; teeCout_ = nullptr;
    delete teeCerr_; teeCerr_ = nullptr;
    if (file_.is_open()) file_.close();
}


// ----------------------------- helpers ------------------------------

void MacMonitor::buildFileNames(const MacConfig& cfg) {
    // timestamp
    std::time_t tt = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    char tsBuf[32];
    std::strftime(tsBuf, sizeof(tsBuf), "%Y%m%d-%H%M%S", &tm);

    const std::string datasetTag = datasetName_.empty() ? std::string("") : (datasetName_ + "__");
    const std::string stem = datasetTag + baseName(cfg.cloudSrcPath) + "__" + baseName(cfg.cloudTgtPath) + "__" + tsBuf;

    logPath_ = outDir_ + "/" + stem + ".log";
    csvPath_ = outDir_ + "/" + stem + ".timings.csv";
}

void MacMonitor::openCsv() {
    csv_.open(csvPath_, std::ios::out);
    if (!csv_.is_open()) {
        std::cerr << "[WARN] Cannot open timing CSV: " << csvPath_ << "\n";
    }
}

void MacMonitor::writeCsvHeader() {
    if (!csv_.is_open()) return;
    // currentS kept for backward naming compatibility (= inclusiveS)
    // csv_ << "level,path,stage,currentS,inclusiveS,exclusiveS,totalS_wall,totalS_stages,rssMb,peakRssMb\n";
    csv_ << "level,path,stage,currentS,inclusiveS,exclusiveS,totalS_wall,totalS_stages,rssMb,peakRssMb,note\n";

    csv_.flush();
}

void MacMonitor::appendCsvRow(int level,
                              const std::string& pathJoined,
                              const std::string& leafStage,
                              double inclusiveS,
                              double exclusiveS,
                              double totalWallS,
                              double totalStagesS,
                              double rssMb,
                              double peakRssMb,
                              const std::string& note) {
    if (!csv_.is_open()) return;

    const double currentS = inclusiveS; // alias

    std::string noteCsv = note;
    for (size_t pos = 0; (pos = noteCsv.find('"', pos)) != std::string::npos; pos += 2) {
        noteCsv.insert(pos, 1, '"'); // " -> ""
    }

    csv_ << level << ","
         << pathJoined << ","
         << leafStage << ","
         << currentS << ","
         << inclusiveS << ","
         << exclusiveS << ","
         << totalWallS << ","
         << totalStagesS << ","
         << rssMb << ","
         << peakRssMb << ","
         << '"' << noteCsv << '"' << "\n"; // <-- quoted
    csv_.flush();
}

std::pair<double,double> MacMonitor::sampleMemoryMb() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(),
                             reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc),
                             sizeof(pmc))) {
        const double rssMb  = static_cast<double>(pmc.WorkingSetSize)     / (1024.0 * 1024.0);
        const double peakMb = static_cast<double>(pmc.PeakWorkingSetSize) / (1024.0 * 1024.0);
        return {rssMb, peakMb};
    }
    return {0.0, 0.0};
#elif defined(__linux__)
    // Peak RSS via getrusage (KiB)
    struct rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    const double peakKb = static_cast<double>(ru.ru_maxrss);
    const double peakMb = peakKb / 1024.0;

    // Current RSS via /proc/self/status -> VmRSS
    double rssMb = 0.0;
    if (FILE* f = std::fopen("/proc/self/status", "r")) {
        char line[256];
        while (std::fgets(line, sizeof(line), f)) {
            if (std::strncmp(line, "VmRSS:", 6) == 0) {
                long kb = 0;
                if (std::sscanf(line + 6, "%ld", &kb) == 1) {
                    rssMb = static_cast<double>(kb) / 1024.0;
                }
                break;
            }
        }
        std::fclose(f);
    }
    return {rssMb, peakMb};
#else
    return {0.0, 0.0};
#endif
}

std::string MacMonitor::baseName(const std::string& path) {
    const auto pos = path.find_last_of("/\\");
    return (pos == std::string::npos) ? path : path.substr(pos + 1);
}

std::string MacMonitor::joinPath(const std::vector<std::string>& parts, char sep) {
    if (parts.empty()) return {};
    std::string out = parts[0];
    for (size_t i = 1; i < parts.size(); ++i) {
        out.push_back(sep);
        out += parts[i];
    }
    return out;
}

// ================== MacMonitor private static helpers (class-scope) ==================
// Build a minimal JSON-safe string by escaping backslash and double-quote.
// Used to compose the compact JSON saved to the CSV "note" column.
std::string MacMonitor::jsonEscape(const std::string& s) {
    std::string out; out.reserve(s.size() + 8);
    for (char c : s) {
        if (c == '\\' || c == '"') { out.push_back('\\'); out.push_back(c); }
        else out.push_back(c);
    }
    return out;
}

// Merge kv-pairs and a free-form note into a compact JSON.
// Numeric-looking values are emitted as numbers; others are quoted.
std::string MacMonitor::buildNoteJson(
    const std::vector<std::pair<std::string, std::string>>& kvs,
    const std::string& extraNote)
{
    std::ostringstream oss;
    oss << '{';
    bool first = true;
    for (const auto&[fst, snd] : kvs) {
        if (!first) oss << ',';
        first = false;
        const std::string& k = fst;
        const std::string& v = snd;
        const bool numeric = !v.empty() && (std::isdigit(v[0]) || v == "true" || v == "false" || v == "null"
                                            || (v[0]=='-' && v.size()>1 && std::isdigit(v[1])));
        oss << '"' << jsonEscape(k) << "\":";
        if (numeric) oss << v;
        else oss << '"' << jsonEscape(v) << '"';
    }
    if (!extraNote.empty()) {
        if (!first) oss << ',';
        oss << "\"_note\":\"" << jsonEscape(extraNote) << '"';
    }
    oss << '}';
    return oss.str();
}

// Pretty tree branch prefix for the time summary (Unicode/ascii).
std::string MacMonitor::treePrefix(const int depth, const bool isLast, const bool useUnicode) {
    if (depth <= 0) return {};
    std::ostringstream oss;
    for (int i = 1; i < depth; ++i) {
        oss << (useUnicode ? "│  " : "|  ");
    }
    oss << (useUnicode ? (isLast ? "└─ " : "├─ ") : (isLast ? "`- " : "|- "));
    return oss.str();
}
// =====================================================================

// ----------------------------- MacMonitor ------------------------------

MacMonitor::MacMonitor() = default;
MacMonitor::~MacMonitor() { shutdown(); }

void MacMonitor::setProgramStart(std::chrono::high_resolution_clock::time_point ts) noexcept {
    programStart_ = ts;
    programStartSet_ = true;
    totalStagesSeconds_ = 0.0;
}

void MacMonitor::init(const MacConfig& cfg) {
    datasetName_ = cfg.currentDatasetName;  // datasetToRun
    srcBase_     = baseName(cfg.cloudSrcPath);
    tgtBase_     = baseName(cfg.cloudTgtPath);
    outDir_      = cfg.outputPath;
    noLogFile_   = cfg.flagNoLogs;

    buildFileNames(cfg);

    // Cache runtime strategy from config (members are declared in MacMonitor.hpp)
    monitorEnabled_     = cfg.monitor.enabled;
    granularity_        = cfg.monitor.granularity;   // CORE / ALL
    coreDepth_          = cfg.monitor.coreDepth;     // only for CORE
    metricsPolicy_      = cfg.monitor.metrics;       // OFF / AUTO / ALL
    timeTreeEnabled_    = cfg.monitor.timeTree.enabled;
    timeTreeUnicode_    = (cfg.monitor.timeTree.charset == MonitorTreeCharset::UNICODE);
    timeTreeShowMemory_ = cfg.monitor.timeTree.showMemory;
    timeTreeShowNotes_  = cfg.monitor.timeTree.showNotes;
    timeTreeMaxDepth_   = cfg.monitor.timeTree.maxDepth;
    evalEnabled_        = cfg.evaluationEnabled;

    // Create output directory if needed (safe no-op if exists)
    if (!outDir_.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(outDir_, ec);
    }

    // If monitoring is globally disabled, do nothing further.
    if (!monitorEnabled_) {
        return;
    }

    // (1) optional tee to file
    tee_.enable(logPath_, noLogFile_);

    // (2) CSV open + header
    openCsv();
    writeCsvHeader();
}

void MacMonitor::shutdown() {
    if (!stack_.empty()) {
        std::cerr << "[WARN] MacMonitor shutdown with " << stack_.size()
                  << " unpaired stage(s). Popping without recording.\n";
        debugDumpOpenStages();
        cancelAll();
    }

    tee_.disable();

    if (csv_.is_open()) {
        csv_.flush();
        csv_.close();
    }
}

void MacMonitor::enter(const std::string& stageName) {
    if (!monitorEnabled_) return; // master switch

    Frame f;
    f.name = stageName;
    f.start = std::chrono::high_resolution_clock::now();
    f.childrenInclusiveSum = 0.0;

    // Depth (1-based) and recordable flag (CORE vs ALL)
    f.level = static_cast<int>(stack_.size()) + 1;
    bool recordThis = true;
    if (granularity_ == MonitorGranularity::CORE) {
        recordThis = (f.level <= std::max(1, coreDepth_));
    }
    f.recordable = recordThis;

    // Prepare metrics buffers
    f.noteText.clear();
    f.kvPairs.clear();

    stack_.push_back(std::move(f));
}

void MacMonitor::record() {
    if (!monitorEnabled_) return; // master switch (align with enter()/setKV()/setNote())
    if (stack_.empty()) {
        std::cerr << "[WARN] MacMonitor::record() called with empty stack. Ignored.\n";
        return;
    }

    const auto now = std::chrono::high_resolution_clock::now();

    // Stop current frame timing; compute inclusiveS
    Frame cur = std::move(stack_.back());
    stack_.pop_back();

    using dur = std::chrono::duration<double>;
    const double inclusiveS = std::chrono::duration_cast<dur>(now - cur.start).count();
    const double exclusiveS = std::max(0.0, inclusiveS - cur.childrenInclusiveSum);

    // Update parent's childrenInclusiveSum (if parent exists)
    if (!stack_.empty()) {
        stack_.back().childrenInclusiveSum += inclusiveS;
    }

    // Update totals (pure algorithm curve)
    totalStagesSeconds_ += inclusiveS;

    // Wall-clock baseline
    double totalWallS = 0.0;
    if (programStartSet_) {
        totalWallS = std::chrono::duration_cast<dur>(now - programStart_).count();
    }

    // Memory sampling (outside timing)
    const auto [rssMb, peakMb] = sampleMemoryMb();

    // Compose CSV row
    const int level = static_cast<int>(stack_.size()) + 1; // depth at the moment of this record
    // Build path parts from remaining stack + current leaf (cur.name)
    std::vector<std::string> parts;
    parts.reserve(stack_.size() + 1);
    for (const auto& fr : stack_) parts.push_back(fr.name);
    parts.push_back(cur.name);
    const std::string pathJoined = joinPath(parts, '/');

    // Metrics -> JSON note  (OFF/AUTO/ALL, with evaluation gating for AUTO)
    const bool allowMetrics = (metricsPolicy_ == MonitorMetrics::ALL) ||
                              (metricsPolicy_ == MonitorMetrics::AUTO && evalEnabled_);
    const std::string noteJson = allowMetrics
        ? buildNoteJson(cur.kvPairs, cur.noteText)
        : buildNoteJson(/*kvs=*/{}, cur.noteText);

    if (cur.recordable) {
        appendCsvRow(level, pathJoined, cur.name,
                     /*inclusiveS*/ inclusiveS,
                     /*exclusiveS*/ exclusiveS,
                     totalWallS, totalStagesSeconds_, rssMb, peakMb,
                     /*note*/ noteJson);
    }

    // Collect for time-tree (only if recordable)
    if (cur.recordable) {
        Recorded rec;
        rec.level      = level;
        rec.name       = cur.name;
        rec.inclusiveS = inclusiveS;
        rec.exclusiveS = exclusiveS;
        rec.rssMB      = rssMb;
        rec.peakMB     = peakMb;
        rec.noteJson   = noteJson;
        recorded_.push_back(std::move(rec));
    }

    // Lightweight console line
    // std::cout << "[MON] <<< " << cur.name
    //           << " | incl=" << std::fixed << std::setprecision(3) << inclusiveS << "s"
    //           << ", excl=" << std::fixed << std::setprecision(3) << exclusiveS << "s"
    //           << (rssMb > 0.0 ? (std::string(" | rss=") + std::to_string(static_cast<int>(rssMb)) + "MB") : std::string())
    //           << "\n";

    // Realtime one-line timing summary through your LOG macro (goes to std::cout)
        std::ostringstream os;
        os << "[MON] " << cur.name
           << " | incl=" << std::fixed << std::setprecision(3) << inclusiveS << "s"
           << ", excl=" << std::fixed << std::setprecision(3) << exclusiveS << "s"
           << (rssMb > 0.0 ? (std::string(" | rss=") + std::to_string(static_cast<int>(rssMb)) + "MB") : std::string());

        LOG_TIMER(os.str());


}

void MacMonitor::cancel() {
    if (stack_.empty()) {
        std::cerr << "[WARN] MacMonitor::cancel() called with empty stack. Ignored.\n";
        return;
    }
    stack_.pop_back(); // Pop silently without writing a row
}

void MacMonitor::cancelAll() {
    stack_.clear();
}

void MacMonitor::recordWholeSession(double totalSeconds) {
    // Synthetic row, level=1, path="wholeSession". Does not add to totalStagesSeconds_.
    double totalWallS = 0.0;
    if (programStartSet_) {
        const auto now = std::chrono::high_resolution_clock::now();
        using dur = std::chrono::duration<double>;
        totalWallS = std::chrono::duration_cast<dur>(now - programStart_).count();
    }
    const auto [rssMb, peakMb] = sampleMemoryMb();

    appendCsvRow(/*level*/1,
                 /*path*/"wholeSession",
                 /*leaf*/"wholeSession",
                 /*inclusiveS*/ totalSeconds,
                 /*exclusiveS*/ totalSeconds, // whole-session row is flat; set the same
                 totalWallS, totalStagesSeconds_, rssMb, peakMb,
                 /*note*/ "{}");
}


void MacMonitor::setNote(const std::string& note) {
    if (!monitorEnabled_) return;
    if (stack_.empty()) {
        std::cerr << "[WARN] MacMonitor::setNote() with empty stack. Ignored.\n";
        return;
    }
    stack_.back().noteText = note;
}

void MacMonitor::clearNote() {
    if (!monitorEnabled_) return;
    if (stack_.empty()) return;
    stack_.back().noteText.clear();
    stack_.back().kvPairs.clear();
}

void MacMonitor::setKV(const std::string& key, const std::string& value) {
    if (!monitorEnabled_) return;
    if (stack_.empty()) {
        std::cerr << "[WARN] MacMonitor::setKV() with empty stack. Ignored.\n";
        return;
    }
    auto& kvs = stack_.back().kvPairs;
    for (auto& kv : kvs) {
        if (kv.first == key) { kv.second = value; return; } // replace
    }
    kvs.emplace_back(key, value); // append
}

void MacMonitor::setKV(const std::string& key, double value) {
    std::ostringstream oss; oss << std::setprecision(12) << value;
    setKV(key, oss.str());
}
void MacMonitor::setKV(const std::string& key, int value) {
    setKV(key, std::to_string(value));
}
void MacMonitor::setKV(const std::string& key, bool value) {
    setKV(key, std::string(value ? "true" : "false"));
}

void MacMonitor::dumpTimeTree() const {
    if (!monitorEnabled_ || !timeTreeEnabled_) return;
    if (recorded_.empty()) {
        std::cout << "[MON] (time-tree) No recorded stages.\n";
        return;
    }

    const bool useUnicode = timeTreeUnicode_;
    const int  maxDepth   = timeTreeMaxDepth_;

    std::cout << "[MON] ===== Hierarchical Time Summary =====\n";

    // Count nodes per level to decide isLast.
    std::vector<int> levelCount;
    std::vector<int> levelVisited;

    {
        std::vector<int> tmp;
        for (const auto& r : recorded_) {
            if (r.level <= 0) continue;
            if (r.level > static_cast<int>(tmp.size())) tmp.resize(r.level, 0);
            tmp[r.level - 1] += 1;
        }
        levelCount = tmp;
        levelVisited.assign(levelCount.size(), 0);
    }

    for (const auto& r : recorded_) {
        const int lvl = r.level;
        if (lvl <= 0) continue;
        if (maxDepth >= 0 && lvl > maxDepth) continue;

        const bool isLast = (++levelVisited[lvl - 1] == levelCount[lvl - 1]);

        std::ostringstream line;
        line << treePrefix(lvl, isLast, useUnicode)
             << r.name
             << "  " << std::fixed << std::setprecision(3) << r.inclusiveS << "s"
             << " (excl " << std::fixed << std::setprecision(3) << r.exclusiveS << "s)";

        if (timeTreeShowMemory_ && r.rssMB > 0.0) {
            line << "  [rss " << static_cast<int>(r.rssMB) << "MB]";
        }
        if (timeTreeShowNotes_ && !r.noteJson.empty()) {
            constexpr size_t kMaxNote = 96;
            if (r.noteJson.size() > kMaxNote) {
                line << "  {" << r.noteJson.substr(1, kMaxNote) << "...";
            } else {
                line << "  " << r.noteJson;
            }
        }
        std::cout << line.str() << "\n";
    }

    std::cout << "[MON] ===== End of Time Summary =====\n";
}

void MacMonitor::debugDumpOpenStages() const noexcept {
    if (stack_.empty()) {
        std::cerr << "[MON][DEBUG] Open stage stack is empty.\n";
        return;
    }
    std::cerr << "[MON][DEBUG] Open stages (root -> leaf):\n";
    for (size_t i = 0; i < stack_.size(); ++i) {
        std::cerr << "  #" << (i+1) << "  " << stack_[i].name << "\n";
    }
}
//
