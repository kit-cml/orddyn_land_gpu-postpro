#ifndef GPU_OPERATIONS_HPP
#define GPU_OPERATIONS_HPP

#include "../modules/glob_type.hpp"
#include "../modules/param.hpp"

/**
 * @brief Prepares GPU memory space and copies initial data from host to device.
 *
 * @param d_ALGEBRAIC Pointer to device memory for algebraic variables.
 * @param ORd_num_of_algebraic Number of algebraic variables.
 * @param sample_size Number of samples.
 * @param d_CONSTANTS Pointer to device memory for constants.
 * @param ORd_num_of_constants Number of constants.
 * @param d_RATES Pointer to device memory for rates.
 * @param ORd_num_of_rates Number of rates.
 * @param d_STATES Pointer to device memory for states.
 * @param ORd_num_of_states Number of states.
 * @param d_p_param Pointer to device memory for parameters.
 * @param temp_result Pointer to device memory for temporary results.
 * @param cipa_result Pointer to device memory for CIPA results.
 * @param d_STATES_RESULT Pointer to device memory for states results.
 * @param d_ic50 Pointer to device memory for IC50 data.
 * @param ic50 Pointer to host memory for IC50 data.
 * @param d_conc Pointer to device memory for concentration data.
 * @param conc Pointer to host memory for concentration data.
 * @param p_param Pointer to host memory for parameters.
 */
void prepingGPUMemory(int sample_size, double *&d_ALGEBRAIC, double *&d_CONSTANTS, double *&d_RATES, double *&d_STATES,
                      double *&d_mec_ALGEBRAIC, double *&d_mec_CONSTANTS, double *&d_mec_RATES, double *&d_mec_STATES,
                      param_t *&d_p_param, cipa_t *&temp_result, cipa_t *&cipa_result, double *&d_STATES_RESULT, double *&d_ic50, 
                      
                      double *ic50, double *&d_conc, double *conc, param_t *p_param, double *&d_herg, double *herg);

/**
 * @brief Frees allocated memory on both the host and device.
 *
 * @param d_ALGEBRAIC Pointer to device memory for algebraic variables.
 * @param d_CONSTANTS Pointer to device memory for constants.
 * @param d_RATES Pointer to device memory for rates.
 * @param d_STATES Pointer to device memory for states.
 * @param d_p_param Pointer to device memory for parameters.
 * @param temp_result Pointer to device memory for temporary results.
 * @param cipa_result Pointer to device memory for CIPA results.
 * @param d_STATES_RESULT Pointer to device memory for states results.
 * @param d_ic50 Pointer to device memory for IC50 data.
 * @param ic50 Pointer to host memory for IC50 data.
 * @param conc Pointer to host memory for concentration data.
 * @param h_states Pointer to host memory for states.
 * @param h_cipa_result Pointer to host memory for CIPA results.
 * @param p_param Pointer to host memory for parameters.
 */
void freeingMemory(double *d_ALGEBRAIC, double *d_CONSTANTS, double *d_RATES, double *d_STATES, double *d_mec_ALGEBRAIC,
                   double *d_mec_CONSTANTS, double *d_mec_RATES, double *d_mec_STATES, param_t *d_p_param,
                   cipa_t *temp_result, cipa_t *cipa_result, double *d_STATES_RESULT, double *d_ic50, double *ic50,
                   double *conc, double *h_states, cipa_t *h_cipa_result, param_t *p_param);

/**
 * @brief Checks the available GPU memory.
 *
 * @param datasize Size of the data to be checked against available GPU memory.
 * @return int 0 if successful, 1 if insufficient memory.
 */
int gpu_check(unsigned int datasize);


void prepingGPUMemoryPostpro(int sample_size, double *&d_ALGEBRAIC, double *&d_CONSTANTS, double *&d_RATES, double *&d_STATES, double *d_STATES_cache,
                      double *&d_mec_ALGEBRAIC, double *&d_mec_CONSTANTS, double *&d_mec_RATES, double *&d_mec_STATES,
                      param_t *&d_p_param, cipa_t *&temp_result, cipa_t *&cipa_result, double *&d_STATES_RESULT, double *&d_ic50,
                      double *ic50, double *&d_cvar, double *cvar, double *&d_conc, double *conc, double *&d_herg, double *herg, param_t *p_param, double *cache,
                      double *time, double *dt, double *states, double *ical, double *inal, double *cai_result, double *ina, double *ito, double *ikr, double *iks, double *ik1, double *tension);

#endif  // GPU_OPERATIONS_HPP
