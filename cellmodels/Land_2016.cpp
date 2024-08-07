#include "Land_2016.hpp"
// #include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "../modules/glob_funct.hpp"
#include "cellmodel.hpp"

// __device__ void initConsts()
// {

// }

// void Land_2016::computeRates()
// {

// }

__device__ inline double check_max(double a, double b) { return fmax(a, b); }

__device__ inline double check_min(double a, double b) { return fmin(a, b); }

__device__ void land_initConsts(bool is_skinned, bool BETA, double *y, double *CONSTANTS, double *RATES, double *STATES,
                                double *ALGEBRAIC, int offset) {
    int num_of_constants = 29;
    int num_of_states = 7;
    int num_of_algebraic = 24;
    int num_of_rates = 7;

    // user input
    CONSTANTS[(offset * num_of_constants) + dlambda_dt] = 0;
    CONSTANTS[(offset * num_of_constants) + lambda] = 1.0;
    CONSTANTS[(offset * num_of_constants) + Cai] = 0.0;
    // if (t == -1) {
    RATES[(offset * num_of_rates) + XS] = 0;
    RATES[(offset * num_of_rates) + XW] = 0;
    RATES[(offset * num_of_rates) + TRPN] = 0;
    RATES[(offset * num_of_rates) + TmBlocked] = 1;
    RATES[(offset * num_of_rates) + ZETAS] = 0;
    RATES[(offset * num_of_rates) + ZETAW] = 0;
    RATES[(offset * num_of_rates) + dCd_dt] = 0;
    //     return;
    // }

    if (CONSTANTS[(offset * num_of_constants) + lambda] >= 1.2) {
        CONSTANTS[(offset * num_of_constants) + lambda] = 1.2;
    }
    CONSTANTS[(offset * num_of_constants) + perm50] = 0.35;
    CONSTANTS[(offset * num_of_constants) + TRPN_n] = 2;
    CONSTANTS[(offset * num_of_constants) + koff] = 0.1;
    CONSTANTS[(offset * num_of_constants) + dr] = 0.25;
    CONSTANTS[(offset * num_of_constants) + wfrac] = 0.5;

    CONSTANTS[(offset * num_of_constants) + TOT_A] = 25;
    CONSTANTS[(offset * num_of_constants) + ktm_unblock] = 0.04;

    CONSTANTS[(offset * num_of_constants) + beta_1] = -2.4;
    CONSTANTS[(offset * num_of_constants) + beta_0] = 2.3;
    CONSTANTS[(offset * num_of_constants) + gamma_idx] = 0.0085;
    CONSTANTS[(offset * num_of_constants) + gamma_wu] = 0.615;
    CONSTANTS[(offset * num_of_constants) + phi] = 2.23;

    if (is_skinned == true) {
        CONSTANTS[(offset * num_of_constants) + nperm] = 2.2;
        ALGEBRAIC[(offset * num_of_algebraic) + ca50] = 2.5;
        CONSTANTS[(offset * num_of_constants) + Tref] = 40.5;
        CONSTANTS[(offset * num_of_constants) + nu] = 1;
        CONSTANTS[(offset * num_of_constants) + mu] = 1;
    } else {
        CONSTANTS[(offset * num_of_constants) + nperm] = 2.4;
        ALGEBRAIC[(offset * num_of_algebraic) + ca50] = 0.805;
        CONSTANTS[(offset * num_of_constants) + Tref] = 120.0;
        CONSTANTS[(offset * num_of_constants) + nu] = 7;
        CONSTANTS[(offset * num_of_constants) + mu] = 3;
    }
    if (BETA == true) {
        // // input an array of beta,
        // CONSTANTS[(offset *  num_of_constants) + beta_1] = beta[1];
        // CONSTANTS[(offset *  num_of_constants) + beta_0] = beta[0];
    }

    // k_ws = 0.004 * mu;
    // k_uw = 0.026  * nu;
    CONSTANTS[(offset * num_of_constants) + k_ws] = 0.004 * CONSTANTS[(offset * num_of_constants) + mu];
    CONSTANTS[(offset * num_of_constants) + k_uw] = 0.026 * CONSTANTS[(offset * num_of_constants) + nu];

    // STATES Variables
    STATES[(offset * num_of_states) + XS] = check_max(0, y[0]);
    STATES[(offset * num_of_states) + XW] = check_max(0, y[1]);
    if (y[2] > 0) STATES[(offset * num_of_states) + TRPN] = check_max(0, y[2]);
    STATES[(offset * num_of_states) + TmBlocked] = y[3];
    STATES[(offset * num_of_states) + ZETAS] = y[4];
    STATES[(offset * num_of_states) + ZETAW] = y[5];
    STATES[(offset * num_of_states) + dCd_dt] = y[6];
    // printf("initialisation\n");

    // seventh state is below, in passive model (dCd_dt)

    CONSTANTS[(offset * num_of_constants) + par_k] = 7;
    CONSTANTS[(offset * num_of_constants) + b] = 9.1;
    CONSTANTS[(offset * num_of_constants) + eta_l] = 200;
    CONSTANTS[(offset * num_of_constants) + eta_s] = 20;
    CONSTANTS[(offset * num_of_constants) + land_a] = 2.1;
}

__device__ void land_computeRates(double TIME, double *CONSTANTS, double *RATES, double *STATES, double *ALGEBRAIC,
                                  double *y, int offset) {
    int num_of_constants = 29;
    int num_of_states = 7;
    int num_of_algebraic = 24;
    int num_of_rates = 7;

    // XB model

    // lambda = min(1.2,lambda);
    // Lfac =  max(0, 1 + beta_0 * (lambda + min(0.87,lambda) - 1.87) );

    CONSTANTS[(offset * num_of_constants) + lambda] = check_min(1.2, CONSTANTS[(offset * num_of_constants) + lambda]);
    ALGEBRAIC[(offset * num_of_algebraic) + Lfac] =
        check_max(0, 1 + CONSTANTS[(offset * num_of_constants) + beta_0] *
                             (CONSTANTS[(offset * num_of_constants) + lambda] +
                              check_min(0.87, CONSTANTS[(offset * num_of_constants) + lambda]) - 1.87));

    if (ALGEBRAIC[(offset * num_of_algebraic) + Lfac] < 0) {
        ALGEBRAIC[(offset * num_of_algebraic) + Lfac] = 0;
    }
    // cdw = phi * k_uw * (1-dr)*(1-wfrac) /  ((1-dr)*wfrac);
    // cds = phi * k_ws * (1-dr)*wfrac / dr;
    ALGEBRAIC[(offset * num_of_algebraic) + cdw] =
        CONSTANTS[(offset * num_of_constants) + phi] * CONSTANTS[(offset * num_of_constants) + k_uw] *
        (1 - CONSTANTS[(offset * num_of_constants) + dr]) * (1 - CONSTANTS[(offset * num_of_constants) + wfrac]) /
        ((1 - CONSTANTS[(offset * num_of_constants) + dr]) * CONSTANTS[(offset * num_of_constants) + wfrac]);
    ALGEBRAIC[(offset * num_of_algebraic) + cds] =
        CONSTANTS[(offset * num_of_constants) + phi] * CONSTANTS[(offset * num_of_constants) + k_ws] *
        (1 - CONSTANTS[(offset * num_of_constants) + dr]) * CONSTANTS[(offset * num_of_constants) + wfrac] /
        CONSTANTS[(offset * num_of_constants) + dr];

    // k_wu = k_uw * (1/wfrac - 1) - k_ws;
    // k_su = k_ws * (1/dr - 1) * wfrac;
    // A = (0.25 * TOT_A) / ((1-dr)*wfrac + dr) * (dr/0.25);
    ALGEBRAIC[(offset * num_of_algebraic) + k_wu] =
        CONSTANTS[(offset * num_of_constants) + k_uw] * (1 / CONSTANTS[(offset * num_of_constants) + wfrac] - 1) -
        CONSTANTS[(offset * num_of_constants) + k_ws];
    ALGEBRAIC[(offset * num_of_algebraic) + k_su] = CONSTANTS[(offset * num_of_constants) + k_ws] *
                                                    (1 / CONSTANTS[(offset * num_of_constants) + dr] - 1) *
                                                    CONSTANTS[(offset * num_of_constants) + wfrac];
    ALGEBRAIC[(offset * num_of_algebraic) + A] =
        (0.25 * CONSTANTS[(offset * num_of_constants) + TOT_A]) /
        ((1 - CONSTANTS[(offset * num_of_constants) + dr]) * CONSTANTS[(offset * num_of_constants) + wfrac] +
         CONSTANTS[(offset * num_of_constants) + dr]) *
        (CONSTANTS[(offset * num_of_constants) + dr] / 0.25);

    // XU   = (1 - TmBlocked) - XW - XS;
    ALGEBRAIC[(offset * num_of_algebraic) + XU] = (1 - STATES[(offset * num_of_states) + TmBlocked]) -
                                                  STATES[(offset * num_of_states) + XW] -
                                                  STATES[(offset * num_of_states) + XS];
    // unattached available xb = all - tm blocked - already prepowerstroke - already post-poststroke - no overlap

    // xb_ws = k_ws * XW;
    // xb_uw = k_uw * XU ;
    // xb_wu = k_wu * XW;
    // xb_su = k_su * XS;
    ALGEBRAIC[(offset * num_of_algebraic) + xb_ws] =
        CONSTANTS[(offset * num_of_constants) + k_ws] * STATES[(offset * num_of_states) + XW];
    ALGEBRAIC[(offset * num_of_algebraic) + xb_uw] =
        CONSTANTS[(offset * num_of_constants) + k_uw] * ALGEBRAIC[(offset * num_of_algebraic) + XU];
    ALGEBRAIC[(offset * num_of_algebraic) + xb_wu] =
        ALGEBRAIC[(offset * num_of_algebraic) + k_wu] * STATES[(offset * num_of_states) + XW];
    ALGEBRAIC[(offset * num_of_algebraic) + xb_su] =
        ALGEBRAIC[(offset * num_of_algebraic) + k_su] * STATES[(offset * num_of_states) + XS];

    // gamma_rate  = gamma * max( (ZETAS > 0) .* ZETAS , (ZETAS < -1) .* (-ZETAS-1) );
    //  element wise multiplication:
    double temp_zetas1, temp_zetas2;
    temp_zetas1 = temp_zetas2 = 0;
    if (STATES[(offset * num_of_states) + ZETAS] > 0) {
        temp_zetas1 = STATES[(offset * num_of_states) + ZETAS];
    }
    if (STATES[(offset * num_of_states) + ZETAS] < -1) {
        temp_zetas2 = -1 * STATES[(offset * num_of_states) + ZETAS] - 1;
    }
    ALGEBRAIC[(offset * num_of_algebraic) + gamma_rate] =
        CONSTANTS[(offset * num_of_constants) + gamma_idx] * check_max(temp_zetas1, temp_zetas2);

    // xb_su_gamma = gamma_rate * XS;
    ALGEBRAIC[(offset * num_of_algebraic) + xb_su_gamma] =
        ALGEBRAIC[(offset * num_of_algebraic) + gamma_rate] * STATES[(offset * num_of_states) + XS];

    // gamma_rate_w  = gamma_wu * abs(ZETAW);
    ALGEBRAIC[(offset * num_of_algebraic) + gamma_rate_w] =
        CONSTANTS[(offset * num_of_constants) + gamma_wu] *
        fabs(STATES[(offset * num_of_states) + ZETAW]);  // weak xbs don't like being strained
    // xb_wu_gamma = gamma_rate_w * XW;
    ALGEBRAIC[(offset * num_of_algebraic) + xb_wu_gamma] =
        ALGEBRAIC[(offset * num_of_algebraic) + gamma_rate_w] * STATES[(offset * num_of_states) + XW];

    // dydt(1)  = xb_ws - xb_su - xb_su_gamma;
    // dydt(2)  = xb_uw - xb_wu - xb_ws - xb_wu_gamma;
    RATES[(offset * num_of_rates) + XS] = ALGEBRAIC[(offset * num_of_algebraic) + xb_ws] -
                                          ALGEBRAIC[(offset * num_of_algebraic) + xb_su] -
                                          ALGEBRAIC[(offset * num_of_algebraic) + xb_wu_gamma];
    RATES[(offset * num_of_rates) + XW] =
        ALGEBRAIC[(offset * num_of_algebraic) + xb_uw] - ALGEBRAIC[(offset * num_of_algebraic) + xb_wu] -
        ALGEBRAIC[(offset * num_of_algebraic) + xb_ws] - ALGEBRAIC[(offset * num_of_algebraic) + xb_wu_gamma];

    // ca50 = ca50 + beta_1*min(0.2,lambda - 1);
    // dydt(3)  = koff * ( (Cai/ca50)^TRPN_n * (1-TRPN) - TRPN);
    ALGEBRAIC[(offset * num_of_algebraic) + ca50] =
        ALGEBRAIC[(offset * num_of_algebraic) + ca50] +
        CONSTANTS[(offset * num_of_constants) + beta_1] *
            check_min(0.2, CONSTANTS[(offset * num_of_constants) + lambda] - 1);
    RATES[(offset * num_of_rates) + TRPN] =
        CONSTANTS[(offset * num_of_constants) + koff] *
        (pow((CONSTANTS[(offset * num_of_constants) + Cai] / ALGEBRAIC[(offset * num_of_algebraic) + ca50]),
             CONSTANTS[(offset * num_of_constants) + TRPN_n]) *
             (1 - STATES[(offset * num_of_states) + TRPN]) -
         STATES[(offset * num_of_states) + TRPN]);

    // XSSS = dr * 0.5;
    // XWSS = (1-dr) * wfrac * 0.5;
    // ktm_block = ktm_unblock * (perm50^nperm) *  0.5 / (0.5 - XSSS - XWSS);
    ALGEBRAIC[(offset * num_of_algebraic) + XSSS] = CONSTANTS[(offset * num_of_constants) + dr] * 0.5;
    ALGEBRAIC[(offset * num_of_algebraic) + XWSS] =
        (1 - CONSTANTS[(offset * num_of_constants) + dr]) * CONSTANTS[(offset * num_of_constants) + wfrac] * 0.5;
    ALGEBRAIC[(offset * num_of_algebraic) + ktm_block] =
        CONSTANTS[(offset * num_of_constants) + ktm_unblock] *
        (pow(CONSTANTS[(offset * num_of_constants) + perm50], CONSTANTS[(offset * num_of_constants) + nperm]) * 0.5) /
        (0.5 - ALGEBRAIC[(offset * num_of_algebraic) + XSSS] - ALGEBRAIC[(offset * num_of_algebraic) + XWSS]);

    // dydt(4)  = ktm_block * min(100, (TRPN^-(nperm/2))) * XU  - ktm_unblock * (TRPN^(nperm/2)) *  TmBlocked;
    RATES[(offset * num_of_rates) + TmBlocked] =
        CONSTANTS[(offset * num_of_constants) + ktm_block] *
            check_min(100, pow(STATES[(offset * num_of_states) + TRPN],
                               -(CONSTANTS[(offset * num_of_constants) + nperm] / 2))) *
            ALGEBRAIC[(offset * num_of_algebraic) + XU] -
        CONSTANTS[(offset * num_of_constants) + ktm_unblock] *
            pow(STATES[(offset * num_of_states) + TRPN], (CONSTANTS[(offset * num_of_constants) + nperm] / 2)) *
            STATES[(offset * num_of_states) + TmBlocked];
    //-------------------------------------------------------------------------------
    // Velocity dependence -- assumes distortion resets on W->S
    // dydt(5) = A * dlambda_dt - cds * ZETAS;% - gamma_rate * ZETAS;
    // dydt(6) = A * dlambda_dt - cdw * ZETAW;% - gamma_rate_w * ZETAW;
    RATES[(offset * num_of_rates) + ZETAS] =
        CONSTANTS[(offset * num_of_constants) + A] * CONSTANTS[(offset * num_of_constants) + dlambda_dt] -
        ALGEBRAIC[(offset * num_of_algebraic) + cds] * STATES[(offset * num_of_states) + ZETAS];
    RATES[(offset * num_of_rates) + ZETAW] =
        CONSTANTS[(offset * num_of_constants) + A] * CONSTANTS[(offset * num_of_constants) + dlambda_dt] -
        ALGEBRAIC[(offset * num_of_algebraic) + cdw] * STATES[(offset * num_of_states) + ZETAW];

    //-------------------------------------------------------------------------------
    // Passive model
    // this is quite scary, did not fix the problem but Cd actually y[6] aka y(7)
    //-------------------------------------------------------------------------------

    // Cd = y(7);
    // C = lambda - 1;

    // CONSTANTS[Cd] = STATES[dCd_dt];
    CONSTANTS[(offset * num_of_constants) + Cd] = y[6];
    CONSTANTS[(offset * num_of_constants) + C] = CONSTANTS[(offset * num_of_constants) + lambda] - 1;

    // if (C - Cd) < 0
    //  eta = eta_s;
    // else
    //  eta = eta_l;
    // end
    if ((CONSTANTS[(offset * num_of_constants) + C] - CONSTANTS[(offset * num_of_constants) + Cd]) < 0) {
        CONSTANTS[(offset * num_of_constants) + eta] = CONSTANTS[(offset * num_of_constants) + eta_s];
    } else {
        CONSTANTS[(offset * num_of_constants) + eta] = CONSTANTS[(offset * num_of_constants) + eta_l];
    }

    // dCd_dt = par_k * (C - Cd) / eta; % F2=Fd
    STATES[(offset * num_of_states) + dCd_dt] =
        CONSTANTS[(offset * num_of_constants) + par_k] *
        (CONSTANTS[(offset * num_of_constants) + C] - CONSTANTS[(offset * num_of_constants) + Cd]) /
        CONSTANTS[(offset * num_of_constants) + eta];
    // dydt(7) = dCd_dt;
    RATES[(offset * num_of_rates) + dCd_dt] = STATES[(offset * num_of_states) + dCd_dt];

    // Fd = eta * dCd_dt;
    ALGEBRAIC[(offset * num_of_algebraic) + Fd] = eta * STATES[(offset * num_of_states) + dCd_dt];
    // F1 = (exp( b * C) - 1);
    ALGEBRAIC[(offset * num_of_algebraic) + F1] =
        (exp(CONSTANTS[(offset * num_of_constants) + b] * CONSTANTS[(offset * num_of_constants) + C]) - 1);
    // Tp = a * (F1 + Fd);
    ALGEBRAIC[(offset * num_of_algebraic) + Tp] =
        CONSTANTS[(offset * num_of_constants) + land_a] *
        (ALGEBRAIC[(offset * num_of_algebraic) + F1] + ALGEBRAIC[(offset * num_of_algebraic) + Fd]);

    //-------------------------------------------------------------------------------
    // Active and Total Force
    //-------------------------------------------------------------------------------

    // Ta = Lfac * (Tref/dr) * ( (ZETAS+1) * XS + (ZETAW) * XW );
    // T = Ta + Tp;
    ALGEBRAIC[(offset * num_of_algebraic) + Ta] =
        ALGEBRAIC[(offset * num_of_algebraic) + Lfac] *
        (CONSTANTS[(offset * num_of_constants) + Tref] / CONSTANTS[(offset * num_of_constants) + dr]) *
        ((STATES[(offset * num_of_states) + ZETAS] + 1) * STATES[(offset * num_of_states) + XS] +
         (STATES[(offset * num_of_states) + ZETAW]) * STATES[(offset * num_of_states) + XW]);
    ALGEBRAIC[(offset * num_of_algebraic) + land_T] =
        ALGEBRAIC[(offset * num_of_algebraic) + Ta] + ALGEBRAIC[(offset * num_of_algebraic) + Tp];
}

__device__ void land_solveEuler(double dt, double t, double Cai_input, double *CONSTANTS, double *RATES, double *STATES,
                                int offset) {
    int num_of_constants = 29;
    int num_of_states = 7;
    int num_of_rates = 7;

    CONSTANTS[(offset * num_of_constants) + Cai] = Cai_input;
    // printf("%lf,%lf\n", t, Cai_input);
    STATES[(offset * num_of_states) + XS] =
        STATES[(offset * num_of_states) + XS] + RATES[(offset * num_of_rates) + XS] * dt;
    STATES[(offset * num_of_states) + XW] =
        STATES[(offset * num_of_states) + XW] + RATES[(offset * num_of_rates) + XW] * dt;
    STATES[(offset * num_of_states) + TRPN] =
        STATES[(offset * num_of_states) + TRPN] + RATES[(offset * num_of_rates) + TRPN] * dt;
    STATES[(offset * num_of_states) + TmBlocked] =
        STATES[(offset * num_of_states) + TmBlocked] + RATES[(offset * num_of_rates) + TmBlocked] * dt;
    STATES[(offset * num_of_states) + ZETAS] =
        STATES[(offset * num_of_states) + ZETAS] + RATES[(offset * num_of_rates) + ZETAS] * dt;
    STATES[(offset * num_of_states) + ZETAW] =
        STATES[(offset * num_of_states) + ZETAW] + RATES[(offset * num_of_rates) + ZETAW] * dt;
    STATES[(offset * num_of_states) + dCd_dt] =
        STATES[(offset * num_of_states) + dCd_dt] + RATES[(offset * num_of_rates) + dCd_dt] * dt;
    // printf("Lfac: %f Ta: %f Tp: %f\n",  ALGEBRAIC[Lfac], ALGEBRAIC[Ta] , ALGEBRAIC[Tp]);
    // karena Lfac 0, jadi Ta nya 0
    // printf("rates: %f %f %f\n",  RATES[XS] , RATES[XW], RATES[TRPN] );
}

// double set_time_step(double TIME, double time_point, double max_time_step, double *CONSTANTS, double *RATES, double
// *STATES, double *ALGEBRAIC, int offset)
// {

// }