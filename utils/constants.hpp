#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include <climits>  // Include this for CHAR_BIT

/**
 * @brief Size of each data point for processing.
 */
const unsigned int datapoint_size = 7000;

/**
 * @brief Maximum number of samples that can be processed.
 */
const unsigned int sample_limit = 10000;

/**
 * @brief Defines a sufficiently large value, calculated using the number of bits in an integer.
 */
const unsigned int ENOUGH = ((CHAR_BIT * sizeof(int) - 1) / 3 + 3);

/**
 * @brief Number of constants, states, algebraic, and rates in the OHara Rudy 2011 model.
 */
const int ORd_num_of_constants = 206;
const int ORd_num_of_states = 50;
const int ORd_num_of_algebraic = 200;
const int ORd_num_of_rates = 50;

/**
 * @brief Number of constants, states, algebraic, and rates in the OHara Rudy 2011 model.
 */
const int Land_num_of_constants = 29;
const int Land_num_of_states = 7;
const int Land_num_of_algebraic = 24;
const int Land_num_of_rates = 7;

/**
 * @brief AUC of INaL and ICaL under control model
 */
const double inal_auc_control = -90.547322;
const double ical_auc_control = -105.935067;

/**
 * @brief Scaling for Calcium and Current
 */
const int CALCIUM_SCALING = 1000000;
const int CURRENT_SCALING = 1000;

/**
 * @brief Number of threads per block for GPU computations.
 */
const int threadsPerBlock = 32;

#endif  // CONSTANTS_HPP
