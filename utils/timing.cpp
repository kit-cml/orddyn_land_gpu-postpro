#include "timing.hpp"
#include <chrono>
#include <iostream>

// A static variable to hold the start time
static std::chrono::time_point<std::chrono::high_resolution_clock> start_time;

/**
 * @brief Starts the timer by recording the current time.
 */
void tic() {
    start_time = std::chrono::high_resolution_clock::now();
}

/**
 * @brief Stops the timer and prints the elapsed time in seconds.
 */
void toc() {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    std::cout << "Elapsed time: " << duration << " s" << std::endl;
}
