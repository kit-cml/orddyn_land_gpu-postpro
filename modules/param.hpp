#ifndef PARAM_HPP
#define PARAM_HPP

/**
 * @brief Structure to hold simulation parameters
 */
struct param_t {
    unsigned short simulation_mode;      ///< Toggle between sample-based or full-pace simulations
    bool is_dutta;                       ///< TRUE if using Dutta scaling
    unsigned short gpu_index;            ///< GPU index for CUDA
    bool is_print_graph;                 ///< TRUE if we want to print graph
    bool is_using_output;                ///< TRUE if using last output file
    bool is_cvar;                        ///< TRUE if using conductance variability
    double bcl;                          ///< Basic cycle length
    unsigned short pace_max;             ///< Maximum pace
    unsigned short find_steepest_start;  ///< Timing to start searching steepest dv/dt repolarization
    unsigned short celltype;             ///< Cell types (0: endo, 1: epi, 2: M)
    double dt;                           ///< Time step
    double dt_write;                     ///< Writing step
    double inet_vm_threshold;            ///< Vm threshold for calculating inet
    char hill_file[1024];                ///< File name for Hill coefficient data
    char cache_file[1024];                ///< File name for in-silico cache data
    char cvar_file[1024];                ///< File name for conductance variability data
    char drug_name[100];                 ///< Name of the drug
    char herg_file[100]; 
    float conc;                          ///< Drug concentration
    int is_time_series;
    int sampling_limit;

    /**
     * @brief Initialize parameters with default values
     */
    void init();

    /**
     * @brief Display the parameter values
     */
    void show_val();
};

/**
 * @brief Structure to hold CiPA simulation results
 */
struct cipa_t {
    double qnet;
    double inal_auc;
    double ical_auc;
    double dvmdt_repol;
    double dvmdt_max;
    double vm_peak;
    double vm_valley;
    double vm_dia;
    double apd90;
    double apd50;
    double apd_tri;
    double ca_peak;
    double ca_valley;
    double ca_dia;
    double cad90;
    double cad50;
    double cad_tri;

    double vm_data[7000];  ///< Membrane potential data
    double vm_time[7000];  ///< Time data for membrane potential

    double dvmdt_data[7000];  ///< Rate of change of membrane potential data
    double dvmdt_time[7000];  ///< Time data for rate of change of membrane potential

    double cai_data[7000];  ///< Intracellular calcium concentration data
    double cai_time[7000];  ///< Time data for intracellular calcium concentration

    double ires_data[7000];  ///< Reserve current data
    double ires_time[7000];  ///< Time data for reserve current

    double inet_data[7000];  ///< Net current data
    double inet_time[7000];  ///< Time data for net current

    double qnet_data[7000];  ///< Net charge data
    double qnet_time[7000];  ///< Time data for net charge
};

#endif  // PARAM_HPP
