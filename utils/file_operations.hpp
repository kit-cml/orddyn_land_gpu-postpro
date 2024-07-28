#ifndef FILE_OPERATIONS_HPP
#define FILE_OPERATIONS_HPP

#include <string>

struct cipa_t;

/**
 * @brief Reads IC50 data and concentration values from a CSV file.
 *
 * @param file_name Name of the file containing IC50 data.
 * @param ic50 Array to store IC50 values.
 * @param conc Array to store concentration values.
 * @return int Number of samples read from the file.
 */
int get_IC50_data_from_file(const char *file_name, double *ic50, double *conc);

/**
 * @brief Reads conductance variability data from a CSV file.
 *
 * @param file_name Name of the file containing conductance variability data.
 * @param limit Maximum number of samples to read.
 * @param cvar Array to store conductance variability data.
 * @return int Number of samples read from the file.
 */
int get_cvar_data_from_file(const char *file_name, unsigned int limit, double *cvar);

/**
 * @brief Extracts the drug name from the given filename.
 *
 * @param filename Name of the file.
 * @return char* Extracted drug name.
 */
char *get_drug_name(const char filename[1024]);

/**
 * @brief Writes simulation results to files in the specified directory.
 *
 * @param base_dir Base directory where results will be stored.
 * @param drug_name Name of the drug being simulated.
 * @param h_states Array of state values from the simulation.
 * @param h_cipa_result Array of CIPA result values from the simulation.
 * @param sample_size Number of samples in the simulation.
 * @param ORd_num_of_states Number of state variables in the simulation.
 */
void write_results_to_file(const char *base_dir, const char *drug_name, double *h_states, cipa_t *h_cipa_result,
                           int sample_size, int ORd_num_of_states);

int get_init_data_from_file(const char* file_name, double *init_states);

int get_herg_data_from_file(const char* dir_name, char* drugname, double *herg);

#endif  // FILE_OPERATIONS_HPP
