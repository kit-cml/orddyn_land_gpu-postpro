#ifndef GPU_CUH
#define GPU_CUH
#include <cuda.h>
#include <cuda_runtime.h>
#include "param.hpp"

__global__ void kernel_DrugSimulation(double *d_ic50, double *d_cvar, double d_conc, double *d_herg, double *d_CONSTANTS, double *d_STATES, double *d_STATES_cache, double *d_RATES, 
                                      double *d_ALGEBRAIC, double *d_mec_CONSTANTS, double *d_mec_STATES, double *d_mec_RATES, double *d_mec_ALGEBRAIC, double *time, double *states, 
                                      double *out_dt, double *cai_result, double *ina, double *inal, double *ical, double *ito, double *ikr, double *iks, double *ik1, double *tension, double *tcurr, 
                                      double *dt, unsigned short sample_id, unsigned int sample_size, cipa_t *temp_result, cipa_t *cipa_result, param_t *p_param
                                      );

__device__ void kernel_DoDrugSim_single(double *d_ic50, double *d_cvar, double *d_conc, double *d_herg, double *d_CONSTANTS,
                                      double *d_STATES, double *d_STATES_cache, double *d_RATES, double *d_ALGEBRAIC,
                                      double *d_mec_CONSTANTS, double *d_mec_STATES, double *d_mec_RATES,
                                      double *d_mec_ALGEBRAIC, double *d_STATES_RESULT, double *d_all_states,
                                      double *time, double *states, double *out_dt, double *cai_result, double *ina,
                                      double *inal, double *ical, double *ito, double *ikr, double *iks, double *ik1, double *tension,
                                      unsigned int sample_size, cipa_t *temp_result, cipa_t *cipa_result, param_t *p_param
                                       );



#endif