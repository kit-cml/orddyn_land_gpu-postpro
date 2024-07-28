#ifndef GLOB_TYPE_HPP
#define GLOB_TYPE_HPP

#include <vector>

/**
 * @brief A structure to hold MPI-related global variables.
 */
struct mympi {
    static char host_name[255]; ///< Host name of the MPI node.
    static int host_name_len;   ///< Length of the host name.
    static int rank;            ///< Rank of the MPI process.
    static int size;            ///< Size of the MPI communicator.
};

/**
 * @brief A structure to hold row data for IC50 values.
 */
struct row_data {
    double data[14]; ///< Array to store 14 features of IC50 data.
};

/**
 * @brief A type alias for a vector of row_data, representing the drug data.
 */
using drug_t = std::vector<row_data>;

/**
 * @brief A structure to store ICaL/INaL control and drug values for calculating qinward.
 */
struct qinward_t {
    double ical_auc_control; ///< ICaL AUC control value (0 concentration).
    double inal_auc_control; ///< INaL AUC control value (0 concentration).
    double ical_auc_drug;    ///< ICaL AUC value with drug.
    double inal_auc_drug;    ///< INaL AUC value with drug.
};

#endif // GLOB_TYPE_HPP
