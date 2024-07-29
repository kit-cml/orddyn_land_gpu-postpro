// #include "cellmodels/enums/enum_Ohara_Rudy_2011.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "../cellmodels/Land_2016.hpp"
#include "../cellmodels/Ohara_Rudy_cipa_v1_2017.hpp"
#include "glob_funct.hpp"
#include "glob_type.hpp"
#include "gpu.cuh"
#include "../utils/constants.hpp"


/*
all kernel function has been moved. Unlike the previous GPU code, now we seperate everything into each modules.
all modules here are optimised for GPU and slightly different than the original component based code
differences are related to GPU offset calculations
*/

__device__ void kernel_DoDrugSim_single(double *d_ic50, double *d_cvar, double *d_conc, double *d_herg, double *d_CONSTANTS,
                                      double *d_STATES, double *d_STATES_cache, double *d_RATES, double *d_ALGEBRAIC,
                                      double *d_mec_CONSTANTS, double *d_mec_STATES, double *d_mec_RATES,
                                      double *d_mec_ALGEBRAIC, double *d_STATES_RESULT, double *d_all_states,
                                      double *time, double *states, double *out_dt, double *cai_result, double *ina,
                                      double *inal, double *ical, double *ito, double *ikr, double *iks, double *ik1, double *tension,
                                      unsigned int sample_size, cipa_t *temp_result, cipa_t *cipa_result, param_t *p_param) {
    unsigned long long input_counter = 0;

    // INIT STARTS
    
    temp_result[sample_id].qnet = 0.;
    temp_result[sample_id].inal_auc = 0.;
    temp_result[sample_id].ical_auc = 0.;
    temp_result[sample_id].dvmdt_repol = -999;
    temp_result[sample_id].dvmdt_max = -999;
    temp_result[sample_id].vm_peak = -999;
    // temp_result[sample_id].vm_valley = d_STATES[(sample_id * ORd_num_of_states) +V];
    temp_result[sample_id].vm_dia = -999;
    temp_result[sample_id].apd90 = 0.;
    temp_result[sample_id].apd50 = 0.;
    temp_result[sample_id].ca_peak = -999;
    // temp_result[sample_id].ca_valley = d_STATES[(sample_id * ORd_num_of_states) +cai];
    temp_result[sample_id].ca_dia = -999;
    temp_result[sample_id].cad90 = 0.;
    temp_result[sample_id].cad50 = 0.;

    cipa_result[sample_id].qnet = 0.;
    cipa_result[sample_id].inal_auc = 0.;
    cipa_result[sample_id].ical_auc = 0.;
    cipa_result[sample_id].dvmdt_repol = -999;
    cipa_result[sample_id].dvmdt_max = -999;
    cipa_result[sample_id].vm_peak = -999;
    // cipa_result[sample_id].vm_valley = d_STATES[(sample_id * ORd_num_of_states) +V];
    cipa_result[sample_id].vm_dia = -999;
    cipa_result[sample_id].apd90 = 0.;
    cipa_result[sample_id].apd50 = 0.;
    cipa_result[sample_id].ca_peak = -999;
    // cipa_result[sample_id].ca_valley = d_STATES[(sample_id * ORd_num_of_states) +cai];
    cipa_result[sample_id].ca_dia = -999;
    cipa_result[sample_id].cad90 = 0.;
    cipa_result[sample_id].cad50 = 0.;
    // INIT ENDS
    bool is_peak = false;
    // to search max dvmdt repol

    tcurr[sample_id] = 0.000001;
    dt[sample_id] = p_param->dt;
    double tmax;
    double max_time_step = 1.0, time_point = 25.0;
    double dt_set;

    int cipa_datapoint = 0;

    // bool writen = false;

    // files for storing results
    // time-series result
    // FILE *fp_vm, *fp_inet, *fp_gate;

    // features
    // double inet, qnet;

    // looping counter
    // unsigned short idx;
  
    // simulation parameters
    // double dtw = 2.0;
    // const char *drug_name = "bepridil";
    // const double bcl = 2000; // bcl is basic cycle length
    const double bcl = p_param->bcl;
    
    const double inet_vm_threshold = p_param->inet_vm_threshold;
    // const unsigned short pace_max = 300;
    // const unsigned short pace_max = 1000;
    const unsigned short pace_max = 2;
    // const unsigned short celltype = 0.;
    // const unsigned short last_pace_print = 3;
    // const unsigned short last_drug_check_pace = 250;
    // const unsigned int print_freq = (1./dt) * dtw;
    // unsigned short pace_count = 0;
    // unsigned short pace_steepest = 0;
    double conc = d_conc; //mmol
    double type = p_param->celltype;
    bool dutta = p_param->is_dutta;
    double epsilon = 10E-14;
    double y[7] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
    // double top_dvmdt = -999.0;

    // eligible AP shape means the Vm_peak > 0.
    bool is_eligible_AP = true;
    // Vm value at 30% repol, 50% repol, and 90% repol, respectively.
    // double vm_repol30, vm_repol50, vm_repol90;
    double t_peak_capture = 0.0;
    unsigned short pace_steepest = 0;

    // qnet_ap/inet_ap values
	  // double inet_ap, qnet_ap, inet4_ap, qnet4_ap, inet_cl, qnet_cl, inet4_cl, qnet4_cl;
	  // double inal_auc_ap, ical_auc_ap,inal_auc_cl, ical_auc_cl;
    // qinward_cl;
     double inet,qinward;
     double inal_auc, ical_auc;
     double vm_repol30, vm_repol50, vm_repol90;
     double t_depol;
     double t_ca_peak, ca_amp50, ca_amp90;
     double cad50_prev, cad50_curr, cad90_prev, cad90_curr;

     //inits
     inal_auc = 0.0; ical_auc = 0.0; inet = 0.0;
     t_ca_peak = 0.0; ca_amp50 = 0.0; ca_amp90 = 0.0;
     vm_repol30 = 999.0; vm_repol50 = 999.0; vm_repol90= 999.0;

    // char buffer[255];

    // static const int CALCIUM_SCALING = 1000000;
	  // static const int CURRENT_SCALING = 1000;

    // printf("Core %d:\n",sample_id);
    // initConsts(d_CONSTANTS, d_STATES, type, conc, d_ic50, d_cvar, dutta, p_param->is_cvar, bcl, sample_id);
    initConsts(d_CONSTANTS, d_STATES, type, conc, d_ic50, d_herg, d_cvar, p_param->is_dutta, p_param->is_cvar, bcl, epsilon, sample_id);
    land_initConsts(false, false, y, d_mec_CONSTANTS, d_mec_RATES, d_mec_STATES, d_mec_ALGEBRAIC, sample_id);

    // starting from initial value, to make things simpler for now, we're just going to replace what initConst has done 
    // to the d_STATES and bring them back to cached initial values:
    for (int temp = 0; temp < ORd_num_of_states; temp++) {
        d_STATES[(sample_id * ORd_num_of_states) + temp] = d_STATES_cache[(sample_id * ORd_num_of_states) + temp];
    }
    
    // these values will follow cache file (instead of regular init)
    temp_result[sample_id].vm_valley = d_STATES[(sample_id * ORd_num_of_states) +V];
    temp_result[sample_id].ca_valley = d_STATES[(sample_id * ORd_num_of_states) +cai];

    cipa_result[sample_id].vm_valley = d_STATES[(sample_id * ORd_num_of_states) +V];
    cipa_result[sample_id].ca_valley = d_STATES[(sample_id * ORd_num_of_states) +cai];

    // temp_result[sample_id].vm_valley = 9.;
    // temp_result[sample_id].ca_valley = 9.;

    // cipa_result[sample_id].vm_valley = 9.;
    // cipa_result[sample_id].ca_valley = 9.;


    // printf("%d: %lf, %d\n", sample_id,d_STATES[V + (sample_id * ORd_num_of_states)], cnt);
    applyDrugEffect(d_CONSTANTS, conc, d_ic50, epsilon, sample_id);

    d_CONSTANTS[BCL + (sample_id * ORd_num_of_constants)] = bcl;

    // generate file for time-series output

    tmax = pace_max * bcl;
    int pace_count = 1;
    
  
    // printf("%d,%lf,%lf,%lf,%lf\n", sample_id, dt[sample_id], tcurr[sample_id], d_STATES[V + (sample_id * ORd_num_of_states)],d_RATES[V + (sample_id * ORd_num_of_rates)]);
    // printf("%lf,%lf,%lf,%lf,%lf\n", d_ic50[0 + (14*sample_id)], d_ic50[1+ (14*sample_id)], d_ic50[2+ (14*sample_id)], d_ic50[3+ (14*sample_id)], d_ic50[4+ (14*sample_id)]);
    while (tcurr[sample_id]<tmax)
    {
        // checked for new coupling algorithm
        land_computeRates(tcurr[sample_id], d_mec_CONSTANTS, d_mec_RATES, d_mec_STATES, d_mec_ALGEBRAIC, y, sample_id);
        computeRates(tcurr[sample_id], d_CONSTANTS, d_RATES, d_STATES, d_ALGEBRAIC, sample_id, d_mec_RATES[TRPN + (sample_id * Land_num_of_rates)]);
        
        
        // dt_set = set_time_step( tcurr[sample_id], time_point, max_time_step, 
        // d_CONSTANTS, 
        // d_RATES, 
        // d_STATES, 
        // d_ALGEBRAIC, 
        // sample_id); 
        dt_set = 0.001;
        if(d_STATES[(sample_id * ORd_num_of_states)+V] > inet_vm_threshold){
          inet += (d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +INaL]+d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +ICaL]+d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +Ito]+d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +IKr]+d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +IKs]+d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +IK1])*dt[sample_id];
          inal_auc += d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +INaL]*dt[sample_id];
          ical_auc += d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +ICaL]*dt[sample_id];
          
          // if (sample_id == 1){
          // printf("%lf %lf %lf %lf %lf %lf\n", 
          // (d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +INaL]+d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +ICaL]+d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +Ito]+d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +IKr]+d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +IKs]+d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +IK1])*dt[sample_id],
          // d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +INaL]*dt[sample_id], 
          // d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +ICaL]*dt[sample_id],
          // inet,
          // inal_auc,
          // ical_auc
          // );
          // }
          
          } 
          // how can we properly update this value?
          // temp_result[sample_id].ca_valley = temp_result[sample_id].cai_data[0];
        
        // printf("tcurr at core %d: %lf\n",sample_id,tcurr[sample_id]);
        if (floor((tcurr[sample_id] + dt_set) / bcl) == floor(tcurr[sample_id] / bcl)) { 
          dt[sample_id] = dt_set;
        }
        else{
          dt[sample_id] = (floor(tcurr[sample_id] / bcl) + 1) * bcl - tcurr[sample_id];

          // new part starts
              // execute at the beginning of a pace
              // temp_result[sample_id].cad50 = cad50_curr - cad50_prev;
              // temp_result[sample_id].cad90 = cad90_curr - cad90_prev; // cad50 and 90 cur not calculcated yet! use outer loop instead
              temp_result[sample_id].qnet = inet/1000.0;
              temp_result[sample_id].inal_auc = inal_auc;
              temp_result[sample_id].ical_auc = ical_auc;
              temp_result[sample_id].vm_dia = d_STATES[(sample_id * ORd_num_of_states)+V];
              temp_result[sample_id].ca_dia = d_STATES[(sample_id * ORd_num_of_states)+cai];

              // cipa_result = temp_result;
              // if(sample_id == 0) printf(" %.2f percent, cipa_result updates!\n", tcurr[sample_id]/tmax);

              cipa_result[sample_id].qnet = temp_result[sample_id].qnet;
              cipa_result[sample_id].inal_auc = temp_result[sample_id].inal_auc;
              cipa_result[sample_id].ical_auc = temp_result[sample_id].ical_auc;
              cipa_result[sample_id].dvmdt_repol = temp_result[sample_id].dvmdt_repol;
              cipa_result[sample_id].dvmdt_max = temp_result[sample_id].dvmdt_max;
              
              cipa_result[sample_id].vm_dia = temp_result[sample_id].vm_dia;
              cipa_result[sample_id].apd90 = temp_result[sample_id].apd90;
              cipa_result[sample_id].apd50 = temp_result[sample_id].apd50;
              cipa_result[sample_id].ca_peak = temp_result[sample_id].ca_peak;
              cipa_result[sample_id].ca_valley = d_STATES[(sample_id * ORd_num_of_states) +cai];
              cipa_result[sample_id].ca_dia = temp_result[sample_id].ca_dia;
              cipa_result[sample_id].cad90 = temp_result[sample_id].cad90;
              cipa_result[sample_id].cad50 = temp_result[sample_id].cad50;
              
              cipa_result[sample_id].dvmdt_repol = temp_result[sample_id].dvmdt_repol;
              cipa_result[sample_id].vm_peak = temp_result[sample_id].vm_peak;
              cipa_result[sample_id].vm_valley = d_STATES[(sample_id * ORd_num_of_states) +V];

              // temp_result[sample_id].ca_valley = d_STATES[(sample_id * ORd_num_of_states) +cai];
              // temp_valley = d_STATES[(sample_id * ORd_num_of_states) +cai];

              // cipa_result[sample_id].qnet_ap = qnet_ap;
              // cipa_result[sample_id].qnet4_ap = qnet4_ap;
              // cipa_result[sample_id].inal_auc_ap = inal_auc_ap;
              // cipa_result[sample_id].ical_auc_ap = ical_auc_ap;
              
              // cipa_result[sample_id].qnet_cl = qnet_cl;
              // cipa_result[sample_id].qnet4_cl = qnet4_cl;
              // cipa_result[sample_id].inal_auc_cl = inal_auc_cl;
              // cipa_result[sample_id].ical_auc_cl = ical_auc_cl;
              
              cipa_result[sample_id].dvmdt_repol = temp_result[sample_id].dvmdt_repol;
              cipa_result[sample_id].vm_peak = temp_result[sample_id].vm_peak;
              cipa_result[sample_id].vm_valley = d_STATES[(sample_id * ORd_num_of_states) +V];
              is_peak = true;
            
          // resetting inet and AUC values
          // and increase the pace count. UPDATE: Disabled since it is very obvious we are using only one pacing here

          // pace_count++;


          input_counter = 0; // at first, we reset the input counter since we re gonna only take one, but I remember we don't have this kind of thing previously, so do we need this still?
          cipa_datapoint = 0; // new pace? reset variables related to saving the values,
          inet = 0.;
          inal_auc = 0.;
          ical_auc = 0.;
          // if(pace_count >= pace_max-last_drug_check_pace){
            // temp_result.init( p_cell->STATES[V], p_cell->STATES[cai] );

            // t_ca_peak = tcurr[sample_id];

            t_depol = (d_CONSTANTS[BCL + (sample_id * ORd_num_of_constants)]*pace_count) + d_CONSTANTS[stim_start + (sample_id * ORd_num_of_constants)];
            // if (sample_id == 1) printf("t_depol: %lf\n",t_depol);
            // is_eligible_AP = false;
            is_eligible_AP = true;
          // }
              
          // new part ends
		
          // printf("core: %d pace count: %d t: %lf, steepest: %d, dvmdt_repol: %lf, t_peak: %lf\n",sample_id,pace_count, tcurr[sample_id], pace_steepest, cipa_result[sample_id].dvmdt_repol,t_peak_capture);
          // writen = false;
        }
        solveAnalytical(d_CONSTANTS, d_STATES, d_ALGEBRAIC, d_RATES,  dt[sample_id], sample_id);
        land_solveEuler(dt[sample_id], tcurr[sample_id], d_STATES[cai + (sample_id * ORd_num_of_states)] * 1000.,
                            d_mec_CONSTANTS, d_mec_RATES, d_mec_STATES, sample_id);
        if( temp_result[sample_id].dvmdt_max < d_RATES[(sample_id * ORd_num_of_states)+V] )temp_result[sample_id].dvmdt_max = d_RATES[(sample_id * ORd_num_of_states)+V];
          
          // this part should be
          // "get the peak Vm 6 secs after depolarization (when Na channel just closed after bursting)" 
          //now it has a different if
			    if( tcurr[sample_id] > ((d_CONSTANTS[(sample_id * ORd_num_of_constants) +BCL]*pace_count)+(d_CONSTANTS[(sample_id * ORd_num_of_constants) +stim_start]+2.)) && 
				      tcurr[sample_id] < ((d_CONSTANTS[(sample_id * ORd_num_of_constants) +BCL]*pace_count)+(d_CONSTANTS[(sample_id * ORd_num_of_constants) +stim_start]+10.)) && 
				      abs(d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +INa]) < 1)
          {
            // printf("check 1\n");
            if( d_STATES[(sample_id * ORd_num_of_states) +V] > temp_result[sample_id].vm_peak )
            {
              temp_result[sample_id].vm_peak = d_STATES[(sample_id * ORd_num_of_states) +V];

              if(temp_result[sample_id].vm_peak > 0)
              {
                vm_repol30 = temp_result[sample_id].vm_peak - (0.3 * (temp_result[sample_id].vm_peak - temp_result[sample_id].vm_valley));
                vm_repol50 = temp_result[sample_id].vm_peak - (0.5 * (temp_result[sample_id].vm_peak - temp_result[sample_id].vm_valley));
                vm_repol90 = temp_result[sample_id].vm_peak - (0.9 * (temp_result[sample_id].vm_peak - temp_result[sample_id].vm_valley));
                is_eligible_AP = true;
                t_peak_capture = tcurr[sample_id];
                // printf("check 2\n");
               
              }
              // else is_eligible_AP = false;
            }
			    }
           // these operations will be executed if it's eligible AP and executed at the beginning of repolarization
			    else if( tcurr[sample_id] > ((d_CONSTANTS[(sample_id * ORd_num_of_constants) +BCL]*pace_count)+(d_CONSTANTS[(sample_id * ORd_num_of_constants) +stim_start]+10)) && is_eligible_AP )
          {
            // printf("check 3\n");
            // printf("rates: %lf, dvmdt_repol: %lf\n states: %lf vm30: %lf, vm90: %lf\n",
            // d_RATES[(sample_id * ORd_num_of_rates) +V],
            // temp_result->dvmdt_repol, 
            // d_STATES[(sample_id * ORd_num_of_states) +V],
            // vm_repol30,
            // vm_repol90
            // );
            // check for valley update
            if( d_STATES[(sample_id * ORd_num_of_states) +cai] < temp_result[sample_id].ca_valley ){
              temp_result[sample_id].ca_valley = d_STATES[(sample_id * ORd_num_of_states) +cai] ;
              // printf("ca valley update\n");
            }


				    if( d_RATES[(sample_id * ORd_num_of_rates) +V] > temp_result[sample_id].dvmdt_repol &&
					      d_STATES[(sample_id * ORd_num_of_states) +V] <= vm_repol30 &&
					      d_STATES[(sample_id * ORd_num_of_states) +V] >= vm_repol90 )
              {
					      temp_result[sample_id].dvmdt_repol = d_RATES[(sample_id * ORd_num_of_rates) +V];
                // printf("check 4\n");
				      }
              // get the APD90, APD50, peak calcium, 50% and 90% of amplitude of Calcium, and time of peak calcium
                // if (sample_id == 1) printf("tcurr[1] : %lf\n",tcurr[sample_id]);

                if( vm_repol50 > d_STATES[(sample_id * ORd_num_of_states) +V] && d_STATES[(sample_id * ORd_num_of_states) +V] > vm_repol50-2 ){
                  temp_result[sample_id].apd50 = tcurr[sample_id] - t_depol;
                  //printf("tcurr: %lf t_depol : %lf\n", tcurr[sample_id], t_depol);  
                } 
                if( vm_repol90 > d_STATES[(sample_id * ORd_num_of_states) +V] && d_STATES[(sample_id * ORd_num_of_states) +V] > vm_repol90-2 ){
                  temp_result[sample_id].apd90 = tcurr[sample_id] - t_depol;
                  } 

                if( temp_result[sample_id].ca_peak < d_STATES[(sample_id * ORd_num_of_states)+cai] ){
                  temp_result[sample_id].ca_peak = d_STATES[(sample_id * ORd_num_of_states) +cai];
                  ca_amp50 = temp_result[sample_id].ca_peak - (0.5 * (temp_result[sample_id].ca_peak - temp_result[sample_id].ca_valley));
                  ca_amp90 = temp_result[sample_id].ca_peak - (0.9 * (temp_result[sample_id].ca_peak - temp_result[sample_id].ca_valley));
                  t_ca_peak = tcurr[sample_id];
                  // printf("ca_amp50 = %lf - (0.5 * (%lf - %lf)) = %lf\n",temp_result[sample_id].ca_peak, temp_result[sample_id].ca_peak,temp_result[sample_id].ca_valley, ca_amp50);
                  // printf("ca_amp90 = %lf - (0.9 * (%lf - %lf)) = %lf\n",temp_result[sample_id].ca_peak, temp_result[sample_id].ca_peak,temp_result[sample_id].ca_valley, ca_amp90);
                  }
          }
          

			    // calculate AP shape
			    // if(is_eligible_AP && d_STATES[(sample_id * ORd_num_of_states) +V] > vm_repol90)
          // {
          //   // printf("check 5 (eligible)\n");
          // // inet_ap/qnet_ap under APD.
          // // inet_ap = (d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +INaL]+d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +ICaL]+d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +Ito]+d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +IKr]+d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +IKs]+d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +IK1]);
          // // inet4_ap = (d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +INaL]+d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +ICaL]+d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +IKr]+d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +INa]);
          // // qnet_ap += (inet_ap * dt[sample_id])/1000.;
          // // qnet4_ap += (inet4_ap * dt[sample_id])/1000.;
          // // inal_auc_ap += (d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +INaL]*dt[sample_id]);
          // // ical_auc_ap += (d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +ICaL]*dt[sample_id]);
			    // }
          // inet_ap/qnet_ap under Cycle Length
          // inet_cl = (d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +INaL]+d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +ICaL]+d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +Ito]+d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +IKr]+d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +IKs]+d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +IK1]);
          // inet4_cl = (d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +INaL]+d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +ICaL]+d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +IKr]+d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +INa]);
          // qnet_cl += (inet_cl * dt[sample_id])/1000.;
          // qnet4_cl += (inet4_cl * dt[sample_id])/1000.;
          // inal_auc_cl += (d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +INaL]*dt[sample_id]);
          // ical_auc_cl += (d_ALGEBRAIC[(sample_id * ORd_num_of_algebraic) +ICaL]*dt[sample_id]);

          // save temporary result -> ALL TEMP RESULTS IN, TEMP RESULT != WRITTEN RESULT
          float tolerance = 0.001f;
          if(cipa_datapoint<p_param->sampling_limit && fmodf(tcurr[sample_id], 1.0f) < tolerance){ // temporary solution to limit the datapoint :(
            if(sample_id==0) {printf("%lf\n", tcurr[sample_id]);}
            temp_result[sample_id].cai_data[cipa_datapoint] =  d_STATES[(sample_id * ORd_num_of_states) +cai] ;
            temp_result[sample_id].cai_time[cipa_datapoint] =  tcurr[sample_id];
            // printf("core: %d, cai_data and time:  %lf %lf datapoint: %d\n",
            // sample_id,
            // temp_result[sample_id].cai_data[cipa_datapoint],
            // temp_result[sample_id].cai_time[cipa_datapoint],
            // cipa_datapoint  );

            temp_result[sample_id].vm_data[cipa_datapoint] = d_STATES[(sample_id * ORd_num_of_states) +V];
            temp_result[sample_id].vm_time[cipa_datapoint] = tcurr[sample_id];

            temp_result[sample_id].dvmdt_data[cipa_datapoint] = d_RATES[(sample_id * ORd_num_of_rates) +V];
            temp_result[sample_id].dvmdt_time[cipa_datapoint] = tcurr[sample_id];

            // time series result

            time[input_counter + sample_id] = tcurr[sample_id];
            states[input_counter + sample_id] = d_STATES[V + (sample_id * ORd_num_of_states)];
            
            out_dt[input_counter + sample_id] = d_RATES[V + (sample_id * ORd_num_of_states)];
            
            cai_result[input_counter + sample_id] = d_STATES[(sample_id * ORd_num_of_states) +cai];

            ina[input_counter + sample_id] = d_ALGEBRAIC[INa + (sample_id * ORd_num_of_algebraic)] ;
            inal[input_counter + sample_id] = d_ALGEBRAIC[INaL + (sample_id * ORd_num_of_algebraic)] ;

            ical[input_counter + sample_id] = d_ALGEBRAIC[ICaL + (sample_id * ORd_num_of_algebraic)] ;
            ito[input_counter + sample_id] = d_ALGEBRAIC[Ito + (sample_id * ORd_num_of_algebraic)] ;

            ikr[input_counter + sample_id] = d_ALGEBRAIC[IKr + (sample_id * ORd_num_of_algebraic)] ;
            iks[input_counter + sample_id] = d_ALGEBRAIC[IKs + (sample_id * ORd_num_of_algebraic)] ;

            ik1[input_counter + sample_id] = d_ALGEBRAIC[IK1 + (sample_id * ORd_num_of_algebraic)] ;

            tension[input_counter + sample_id] = d_mec_ALGEBRAIC[land_T + (sample_id * Land_num_of_algebraic)] * 480.0; 

            input_counter = input_counter + sample_size;
            cipa_datapoint = cipa_datapoint + 1; // this causes the resource usage got so mega and crashed in running
          }

          // cipa result update
              cipa_result[sample_id].qnet = temp_result[sample_id].qnet;
              cipa_result[sample_id].inal_auc = temp_result[sample_id].inal_auc;
              cipa_result[sample_id].ical_auc = temp_result[sample_id].ical_auc;
              cipa_result[sample_id].dvmdt_repol = temp_result[sample_id].dvmdt_repol;
              cipa_result[sample_id].dvmdt_max = temp_result[sample_id].dvmdt_max;
              
              cipa_result[sample_id].vm_dia = temp_result[sample_id].vm_dia;
              cipa_result[sample_id].apd90 = temp_result[sample_id].apd90;
              cipa_result[sample_id].apd50 = temp_result[sample_id].apd50;
              cipa_result[sample_id].ca_peak = temp_result[sample_id].ca_peak;
              cipa_result[sample_id].ca_valley = d_STATES[(sample_id * ORd_num_of_states) +cai];
              cipa_result[sample_id].ca_dia = temp_result[sample_id].ca_dia;
              
              
              cipa_result[sample_id].dvmdt_repol = temp_result[sample_id].dvmdt_repol;
              cipa_result[sample_id].vm_peak = temp_result[sample_id].vm_peak;
              cipa_result[sample_id].vm_valley = d_STATES[(sample_id * ORd_num_of_states) +V];

	
        tcurr[sample_id] = tcurr[sample_id] + dt[sample_id];
        //printf("t after addition: %lf\n", tcurr[sample_id]);
       
  }
    // __syncthreads();

    // // looking for cad50 and 90
    for(int ca_looper = 0; ca_looper < p_param->sampling_limit; ca_looper++){
          // before the peak calcium
          
          if( temp_result[sample_id].cai_time[ca_looper] < t_ca_peak ){
            // printf("cai_data %lf \n",temp_result[sample_id].cai_data[ca_looper]);
            if( temp_result[sample_id].cai_data[ca_looper] < ca_amp50 ){
              cad50_prev = temp_result[sample_id].cai_time[ca_looper];
              // printf("cad50 prev update\n");
            } 
            if( temp_result[sample_id].cai_data[ca_looper] < ca_amp90 ){
              cad90_prev = temp_result[sample_id].cai_time[ca_looper];
              // printf("cad90 prev update\n");
            } 
          }
          // after the peak calcium
          else{
            if( temp_result[sample_id].cai_data[ca_looper] > ca_amp50 ) cad50_curr = temp_result[sample_id].cai_time[ca_looper];
            if( temp_result[sample_id].cai_data[ca_looper] > ca_amp90 ) cad90_curr = temp_result[sample_id].cai_time[ca_looper];
          }
        }
      // printf("core: %d ca_peak %lf | : 50: %lf - %lf 90: %lf - %lf\n",sample_id, t_ca_peak, cad50_curr, cad50_prev, cad90_curr, cad90_prev);
      // printf("ca_peak: %lf ca_valley %lf\n", temp_result[sample_id].ca_peak, temp_result[sample_id].ca_valley);
      // printf("cai_data[0] %lf \n",temp_result[sample_id].cai_data[0]);
      temp_result[sample_id].cad50 = cad50_curr - cad50_prev;// the curr is lower than the prev, like waaay lower, its a negative (it shouldnt be, since its in time)
      temp_result[sample_id].cad90 = cad90_curr - cad90_prev;
      cipa_result[sample_id].cad90 = temp_result[sample_id].cad90;
      cipa_result[sample_id].cad50 = temp_result[sample_id].cad50;
}

__global__ void kernel_DrugSimulation(double *d_ic50, double *d_cvar, double d_conc, double *d_herg, double *d_CONSTANTS, double *d_STATES, double *d_STATES_cache, double *d_RATES, 
                                      double *d_ALGEBRAIC, double *d_mec_CONSTANTS, double *d_mec_STATES, double *d_mec_RATES, double *d_mec_ALGEBRAIC, double *time, double *states, 
                                      double *out_dt, double *cai_result, double *ina, double *inal, double *ical, double *ito, double *ikr, double *iks, double *ik1, double *tension, double *tcurr, 
                                      double *dt, unsigned short sample_id, unsigned int sample_size, cipa_t *temp_result, cipa_t *cipa_result, param_t *p_param) {
    unsigned short thread_id;
    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_id >= sample_size){
        printf("too big, detected: %d from sample size of %d\n", thread_id, sample_size);
        return;
    } 
    double time_for_each_sample[10000];
    double dt_for_each_sample[10000];

    kernel_DoDrugSim_single(d_ic50, d_cvar, d_conc[thread_id], d_herg, d_CONSTANTS, d_STATES, d_STATES_cache, d_RATES,
                            d_ALGEBRAIC, d_mec_CONSTANTS, d_mec_STATES, d_mec_RATES, d_mec_ALGEBRAIC, time, states,
                            out_dt, cai_result, ina, inal, ical, ito, ikr, iks, ik1, tension, time_for_each_sample,
                            dt_for_each_sample, thread_id, sample_size, temp_result, cipa_result, p_param);
}