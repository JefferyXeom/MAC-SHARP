//
// Created by Jeffery_Xeom on 2025/8/24.
// Project: MAC_SHARP
// File: MAC_Timer.hpp
//

#pragma once

//
// Created by Jeffery_Xeom on 2025/8/24.
//

#include <chrono>
#include <vector>

#include "MacTimer.hpp"
#include "CommonTypes.hpp" // For logging macros


// Timer class should be carefully checked
/**
 * @class Timer
 * @brief A timer class to measure the elapsed time
 */
class Timer {
    // private
    std::chrono::high_resolution_clock::time_point startTime;
    std::chrono::high_resolution_clock::time_point endTime;
    std::vector<double> elapsedTimes;
    std::string currentSession;
    bool isRunning;

public:
    Timer() : isRunning(false) {}

    void startTiming(const std::string& sessionName) {
        currentSession = sessionName;
        startTime = std::chrono::high_resolution_clock::now();
        isRunning = true;
        LOG_TIMER(sessionName << " started");
    }

    double endTiming() {
        if (!isRunning) {
            LOG_TIMER(RESET << YELLOW << "Error: No active timing session" << RESET); // 临时修改一下这个位置的log
            return 0.0;
        }

        endTime = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double> elapsed = endTime - startTime;
        const double elapsedSeconds = elapsed.count();

        elapsedTimes.push_back(elapsedSeconds);
        isRunning = false;

        LOG_TIMER(currentSession << " completed: " << elapsedSeconds << " seconds");

        return elapsedSeconds;
    }

    [[nodiscard]] const std::vector<double>& getElapsedTimes() const {
        return elapsedTimes;
    }

    [[nodiscard]] double getLastElapsedTime() const {
        return elapsedTimes.empty() ? 0.0 : elapsedTimes.back();
    }

    void clearHistory() {
        elapsedTimes.clear();
    }

    [[nodiscard]] std::string getCurrentSession() const {
        return currentSession;
    }

    [[nodiscard]] bool getIsRunning() const {
        return isRunning;
    }
};
