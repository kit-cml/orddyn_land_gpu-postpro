/*
   There are a total of 200 entries in the algebraic variable array.
   There are a total of 49 entries in each of the rate and state variable arrays.
   There are a total of 206 entries in the constant variable array.
 */

#include "Ohara_Rudy_cipa_v1_2017.hpp"
#include <cmath>
#include <cstdlib>
// #include "../../functions/inputoutput.hpp"
#include <cstdio>
#include "../modules/glob_funct.hpp"
#include "../utils/constants.hpp"
#include <cuda_runtime.h>
#include <cuda.h>

/*
 * TIME is time in component environment (millisecond).
 * CONSTANTS[celltype] is celltype in component environment (dimensionless).
 * CONSTANTS[nao] is nao in component extracellular (millimolar).
 * CONSTANTS[cao] is cao in component extracellular (millimolar).
 * CONSTANTS[ko] is ko in component extracellular (millimolar).
 * CONSTANTS[R] is R in component physical_constants (joule_per_kilomole_kelvin).
 * CONSTANTS[T] is T in component physical_constants (kelvin).
 * CONSTANTS[F] is F in component physical_constants (coulomb_per_mole).
 * CONSTANTS[zna] is zna in component physical_constants (dimensionless).
 * CONSTANTS[zca] is zca in component physical_constants (dimensionless).
 * CONSTANTS[zk] is zk in component physical_constants (dimensionless).
 * CONSTANTS[L] is L in component cell_geometry (centimeter).
 * CONSTANTS[rad] is rad in component cell_geometry (centimeter).
 * CONSTANTS[vcell] is vcell in component cell_geometry (microliter).
 * CONSTANTS[Ageo] is Ageo in component cell_geometry (centimeter_squared).
 * CONSTANTS[Acap] is Acap in component cell_geometry (centimeter_squared).
 * CONSTANTS[vmyo] is vmyo in component cell_geometry (microliter).
 * CONSTANTS[vnsr] is vnsr in component cell_geometry (microliter).
 * CONSTANTS[vjsr] is vjsr in component cell_geometry (microliter).
 * CONSTANTS[vss] is vss in component cell_geometry (microliter).
 * STATES[V] is v in component membrane (millivolt).
 * ALGEBRAIC[vfrt] is vfrt in component membrane (dimensionless).
 * CONSTANTS[ffrt] is ffrt in component membrane (coulomb_per_mole_millivolt).
 * CONSTANTS[frt] is frt in component membrane (per_millivolt).
 * ALGEBRAIC[INa] is INa in component INa (microA_per_microF).
 * ALGEBRAIC[INaL] is INaL in component INaL (microA_per_microF).
 * ALGEBRAIC[Ito] is Ito in component Ito (microA_per_microF).
 * ALGEBRAIC[ICaL] is ICaL in component ICaL (microA_per_microF).
 * ALGEBRAIC[ICaNa] is ICaNa in component ICaL (microA_per_microF).
 * ALGEBRAIC[ICaK] is ICaK in component ICaL (microA_per_microF).
 * ALGEBRAIC[IKr] is IKr in component IKr (microA_per_microF).
 * ALGEBRAIC[IKs] is IKs in component IKs (microA_per_microF).
 * ALGEBRAIC[IK1] is IK1 in component IK1 (microA_per_microF).
 * ALGEBRAIC[INaCa_i] is INaCa_i in component INaCa_i (microA_per_microF).
 * ALGEBRAIC[INaCa_ss] is INaCa_ss in component INaCa_i (microA_per_microF).
 * ALGEBRAIC[INaK] is INaK in component INaK (microA_per_microF).
 * ALGEBRAIC[INab] is INab in component INab (microA_per_microF).
 * ALGEBRAIC[IKb] is IKb in component IKb (microA_per_microF).
 * ALGEBRAIC[IpCa] is IpCa in component IpCa (microA_per_microF).
 * ALGEBRAIC[ICab] is ICab in component ICab (microA_per_microF).
 * ALGEBRAIC[Istim] is Istim in component membrane (microA_per_microF).
 * CONSTANTS[stim_start] is stim_start in component membrane (millisecond).
 * CONSTANTS[stim_end] is stim_end in component membrane (millisecond).
 * CONSTANTS[amp] is amp in component membrane (microA_per_microF).
 * CONSTANTS[BCL] is BCL in component membrane (millisecond).
 * CONSTANTS[duration] is duration in component membrane (millisecond).
 * CONSTANTS[KmCaMK] is KmCaMK in component CaMK (millimolar).
 * CONSTANTS[aCaMK] is aCaMK in component CaMK (per_millimolar_per_millisecond).
 * CONSTANTS[bCaMK] is bCaMK in component CaMK (per_millisecond).
 * CONSTANTS[CaMKo] is CaMKo in component CaMK (dimensionless).
 * CONSTANTS[KmCaM] is KmCaM in component CaMK (millimolar).
 * ALGEBRAIC[CaMKb] is CaMKb in component CaMK (millimolar).
 * ALGEBRAIC[CaMKa] is CaMKa in component CaMK (millimolar).
 * STATES[CaMKt] is CaMKt in component CaMK (millimolar).
 * STATES[cass] is cass in component intracellular_ions (millimolar).
 * CONSTANTS[cmdnmax_b] is cmdnmax_b in component intracellular_ions (millimolar).
 * CONSTANTS[cmdnmax] is cmdnmax in component intracellular_ions (millimolar).
 * CONSTANTS[kmcmdn] is kmcmdn in component intracellular_ions (millimolar).
 * CONSTANTS[trpnmax] is trpnmax in component intracellular_ions (millimolar).
 * CONSTANTS[kmtrpn] is kmtrpn in component intracellular_ions (millimolar).
 * CONSTANTS[BSRmax] is BSRmax in component intracellular_ions (millimolar).
 * CONSTANTS[KmBSR] is KmBSR in component intracellular_ions (millimolar).
 * CONSTANTS[BSLmax] is BSLmax in component intracellular_ions (millimolar).
 * CONSTANTS[KmBSL] is KmBSL in component intracellular_ions (millimolar).
 * CONSTANTS[csqnmax] is csqnmax in component intracellular_ions (millimolar).
 * CONSTANTS[kmcsqn] is kmcsqn in component intracellular_ions (millimolar).
 * STATES[nai] is nai in component intracellular_ions (millimolar).
 * STATES[nass] is nass in component intracellular_ions (millimolar).
 * STATES[ki] is ki in component intracellular_ions (millimolar).
 * STATES[kss] is kss in component intracellular_ions (millimolar).
 * STATES[cansr] is cansr in component intracellular_ions (millimolar).
 * STATES[cajsr] is cajsr in component intracellular_ions (millimolar).
 * STATES[cai] is cai in component intracellular_ions (millimolar).
 * ALGEBRAIC[JdiffNa] is JdiffNa in component diff (millimolar_per_millisecond).
 * ALGEBRAIC[Jdiff] is Jdiff in component diff (millimolar_per_millisecond).
 * ALGEBRAIC[Jup] is Jup in component SERCA (millimolar_per_millisecond).
 * ALGEBRAIC[JdiffK] is JdiffK in component diff (millimolar_per_millisecond).
 * ALGEBRAIC[Jrel] is Jrel in component ryr (millimolar_per_millisecond).
 * ALGEBRAIC[Jtr] is Jtr in component trans_flux (millimolar_per_millisecond).
 * ALGEBRAIC[Bcai] is Bcai in component intracellular_ions (dimensionless).
 * ALGEBRAIC[Bcajsr] is Bcajsr in component intracellular_ions (dimensionless).
 * ALGEBRAIC[Bcass] is Bcass in component intracellular_ions (dimensionless).
 * CONSTANTS[cm] is cm in component intracellular_ions (microF_per_centimeter_squared).
 * CONSTANTS[PKNa] is PKNa in component reversal_potentials (dimensionless).
 * ALGEBRAIC[ENa] is ENa in component reversal_potentials (millivolt).
 * ALGEBRAIC[EK] is EK in component reversal_potentials (millivolt).
 * ALGEBRAIC[EKs] is EKs in component reversal_potentials (millivolt).
 * ALGEBRAIC[mss] is mss in component INa (dimensionless).
 * ALGEBRAIC[tm] is tm in component INa (millisecond).
 * CONSTANTS[mssV1] is mssV1 in component INa (millivolt).
 * CONSTANTS[mssV2] is mssV2 in component INa (millivolt).
 * CONSTANTS[mtV1] is mtV1 in component INa (millivolt).
 * CONSTANTS[mtV2] is mtV2 in component INa (millivolt).
 * CONSTANTS[mtD1] is mtD1 in component INa (dimensionless).
 * CONSTANTS[mtD2] is mtD2 in component INa (dimensionless).
 * CONSTANTS[mtV3] is mtV3 in component INa (millivolt).
 * CONSTANTS[mtV4] is mtV4 in component INa (millivolt).
 * STATES[m] is m in component INa (dimensionless).
 * ALGEBRAIC[hss] is hss in component INa (dimensionless).
 * ALGEBRAIC[thf] is thf in component INa (millisecond).
 * ALGEBRAIC[ths] is ths in component INa (millisecond).
 * CONSTANTS[hssV1] is hssV1 in component INa (millivolt).
 * CONSTANTS[hssV2] is hssV2 in component INa (millivolt).
 * CONSTANTS[Ahs] is Ahs in component INa (dimensionless).
 * CONSTANTS[Ahf] is Ahf in component INa (dimensionless).
 * STATES[hf] is hf in component INa (dimensionless).
 * STATES[hs] is hs in component INa (dimensionless).
 * ALGEBRAIC[h] is h in component INa (dimensionless).
 * CONSTANTS[GNa] is GNa in component INa (milliS_per_microF).
 * CONSTANTS[shift_INa_inact] is shift_INa_inact in component INa (millivolt).
 * ALGEBRAIC[jss] is jss in component INa (dimensionless).
 * ALGEBRAIC[tj] is tj in component INa (millisecond).
 * STATES[j] is j in component INa (dimensionless).
 * ALGEBRAIC[hssp] is hssp in component INa (dimensionless).
 * ALGEBRAIC[thsp] is thsp in component INa (millisecond).
 * STATES[hsp] is hsp in component INa (dimensionless).
 * ALGEBRAIC[hp] is hp in component INa (dimensionless).
 * ALGEBRAIC[tjp] is tjp in component INa (millisecond).
 * STATES[jp] is jp in component INa (dimensionless).
 * ALGEBRAIC[fINap] is fINap in component INa (dimensionless).
 * ALGEBRAIC[mLss] is mLss in component INaL (dimensionless).
 * ALGEBRAIC[tmL] is tmL in component INaL (millisecond).
 * STATES[mL] is mL in component INaL (dimensionless).
 * CONSTANTS[thL] is thL in component INaL (millisecond).
 * ALGEBRAIC[hLss] is hLss in component INaL (dimensionless).
 * STATES[hL] is hL in component INaL (dimensionless).
 * ALGEBRAIC[hLssp] is hLssp in component INaL (dimensionless).
 * CONSTANTS[thLp] is thLp in component INaL (millisecond).
 * STATES[hLp] is hLp in component INaL (dimensionless).
 * CONSTANTS[GNaL_b] is GNaL_b in component INaL (milliS_per_microF).
 * CONSTANTS[GNaL] is GNaL in component INaL (milliS_per_microF).
 * ALGEBRAIC[fINaLp] is fINaLp in component INaL (dimensionless).
 * CONSTANTS[Gto_b] is Gto_b in component Ito (milliS_per_microF).
 * ALGEBRAIC[ass] is ass in component Ito (dimensionless).
 * ALGEBRAIC[ta] is ta in component Ito (millisecond).
 * STATES[a] is a in component Ito (dimensionless).
 * ALGEBRAIC[iss] is iss in component Ito (dimensionless).
 * ALGEBRAIC[delta_epi] is delta_epi in component Ito (dimensionless).
 * ALGEBRAIC[tiF_b] is tiF_b in component Ito (millisecond).
 * ALGEBRAIC[tiS_b] is tiS_b in component Ito (millisecond).
 * ALGEBRAIC[tiF] is tiF in component Ito (millisecond).
 * ALGEBRAIC[tiS] is tiS in component Ito (millisecond).
 * ALGEBRAIC[AiF] is AiF in component Ito (dimensionless).
 * ALGEBRAIC[AiS] is AiS in component Ito (dimensionless).
 * STATES[iF] is iF in component Ito (dimensionless).
 * STATES[iS] is iS in component Ito (dimensionless).
 * ALGEBRAIC[i] is i in component Ito (dimensionless).
 * ALGEBRAIC[assp] is assp in component Ito (dimensionless).
 * STATES[ap] is ap in component Ito (dimensionless).
 * ALGEBRAIC[dti_develop] is dti_develop in component Ito (dimensionless).
 * ALGEBRAIC[dti_recover] is dti_recover in component Ito (dimensionless).
 * ALGEBRAIC[tiFp] is tiFp in component Ito (millisecond).
 * ALGEBRAIC[tiSp] is tiSp in component Ito (millisecond).
 * STATES[iFp] is iFp in component Ito (dimensionless).
 * STATES[iSp] is iSp in component Ito (dimensionless).
 * ALGEBRAIC[ip] is ip in component Ito (dimensionless).
 * CONSTANTS[Gto] is Gto in component Ito (milliS_per_microF).
 * ALGEBRAIC[fItop] is fItop in component Ito (dimensionless).
 * CONSTANTS[Kmn] is Kmn in component ICaL (millimolar).
 * CONSTANTS[k2n] is k2n in component ICaL (per_millisecond).
 * CONSTANTS[PCa_b] is PCa_b in component ICaL (dimensionless).
 * ALGEBRAIC[dss] is dss in component ICaL (dimensionless).
 * STATES[d] is d in component ICaL (dimensionless).
 * ALGEBRAIC[fss] is fss in component ICaL (dimensionless).
 * CONSTANTS[Aff] is Aff in component ICaL (dimensionless).
 * CONSTANTS[Afs] is Afs in component ICaL (dimensionless).
 * STATES[ff] is ff in component ICaL (dimensionless).
 * STATES[fs] is fs in component ICaL (dimensionless).
 * ALGEBRAIC[f] is f in component ICaL (dimensionless).
 * ALGEBRAIC[fcass] is fcass in component ICaL (dimensionless).
 * ALGEBRAIC[Afcaf] is Afcaf in component ICaL (dimensionless).
 * ALGEBRAIC[Afcas] is Afcas in component ICaL (dimensionless).
 * STATES[fcaf] is fcaf in component ICaL (dimensionless).
 * STATES[fcas] is fcas in component ICaL (dimensionless).
 * ALGEBRAIC[fca] is fca in component ICaL (dimensionless).
 * STATES[jca] is jca in component ICaL (dimensionless).
 * STATES[ffp] is ffp in component ICaL (dimensionless).
 * ALGEBRAIC[fp] is fp in component ICaL (dimensionless).
 * STATES[fcafp] is fcafp in component ICaL (dimensionless).
 * ALGEBRAIC[fcap] is fcap in component ICaL (dimensionless).
 * ALGEBRAIC[km2n] is km2n in component ICaL (per_millisecond).
 * ALGEBRAIC[anca] is anca in component ICaL (dimensionless).
 * STATES[nca] is nca in component ICaL (dimensionless).
 * ALGEBRAIC[PhiCaL] is PhiCaL in component ICaL (dimensionless).
 * ALGEBRAIC[PhiCaNa] is PhiCaNa in component ICaL (dimensionless).
 * ALGEBRAIC[PhiCaK] is PhiCaK in component ICaL (dimensionless).
 * CONSTANTS[PCa] is PCa in component ICaL (dimensionless).
 * CONSTANTS[PCap] is PCap in component ICaL (dimensionless).
 * CONSTANTS[PCaNa] is PCaNa in component ICaL (dimensionless).
 * CONSTANTS[PCaK] is PCaK in component ICaL (dimensionless).
 * CONSTANTS[PCaNap] is PCaNap in component ICaL (dimensionless).
 * CONSTANTS[PCaKp] is PCaKp in component ICaL (dimensionless).
 * ALGEBRAIC[fICaLp] is fICaLp in component ICaL (dimensionless).
 * ALGEBRAIC[td] is td in component ICaL (millisecond).
 * ALGEBRAIC[tff] is tff in component ICaL (millisecond).
 * ALGEBRAIC[tfs] is tfs in component ICaL (millisecond).
 * ALGEBRAIC[tfcaf] is tfcaf in component ICaL (millisecond).
 * ALGEBRAIC[tfcas] is tfcas in component ICaL (millisecond).
 * CONSTANTS[tjca] is tjca in component ICaL (millisecond).
 * ALGEBRAIC[tffp] is tffp in component ICaL (millisecond).
 * ALGEBRAIC[tfcafp] is tfcafp in component ICaL (millisecond).
 * CONSTANTS[v0_CaL] is v0 in component ICaL (millivolt).
 * ALGEBRAIC[A_1] is A_1 in component ICaL (dimensionless).
 * CONSTANTS[B_1] is B_1 in component ICaL (per_millivolt).
 * ALGEBRAIC[U_1] is U_1 in component ICaL (dimensionless).
 * ALGEBRAIC[A_2] is A_2 in component ICaL (dimensionless).
 * CONSTANTS[B_2] is B_2 in component ICaL (per_millivolt).
 * ALGEBRAIC[U_2] is U_2 in component ICaL (dimensionless).
 * ALGEBRAIC[A_3] is A_3 in component ICaL (dimensionless).
 * CONSTANTS[B_3] is B_3 in component ICaL (per_millivolt).
 * ALGEBRAIC[U_3] is U_3 in component ICaL (dimensionless).
 * CONSTANTS[GKr_b] is GKr_b in component IKr (milliS_per_microF).
 * STATES[IC1] is IC1 in component IKr (dimensionless).
 * STATES[IC2] is IC2 in component IKr (dimensionless).
 * STATES[C1] is C1 in component IKr (dimensionless).
 * STATES[C2] is C2 in component IKr (dimensionless).
 * STATES[O] is O in component IKr (dimensionless).
 * STATES[IO] is IO in component IKr (dimensionless).
 * STATES[IObound] is IObound in component IKr (dimensionless).
 * STATES[Obound] is Obound in component IKr (dimensionless).
 * STATES[Cbound] is Cbound in component IKr (dimensionless).
 * STATES[D] is D in component IKr (dimensionless).
 * CONSTANTS[GKr] is GKr in component IKr (milliS_per_microF).
 * CONSTANTS[A1] is A1 in component IKr (per_millisecond).
 * CONSTANTS[B1] is B1 in component IKr (per_millivolt).
 * CONSTANTS[q1] is q1 in component IKr (dimensionless).
 * CONSTANTS[A2] is A2 in component IKr (per_millisecond).
 * CONSTANTS[B2] is B2 in component IKr (per_millivolt).
 * CONSTANTS[q2] is q2 in component IKr (dimensionless).
 * CONSTANTS[A3] is A3 in component IKr (per_millisecond).
 * CONSTANTS[B3] is B3 in component IKr (per_millivolt).
 * CONSTANTS[q3] is q3 in component IKr (dimensionless).
 * CONSTANTS[A4] is A4 in component IKr (per_millisecond).
 * CONSTANTS[B4] is B4 in component IKr (per_millivolt).
 * CONSTANTS[q4] is q4 in component IKr (dimensionless).
 * CONSTANTS[A11] is A11 in component IKr (per_millisecond).
 * CONSTANTS[B11] is B11 in component IKr (per_millivolt).
 * CONSTANTS[q11] is q11 in component IKr (dimensionless).
 * CONSTANTS[A21] is A21 in component IKr (per_millisecond).
 * CONSTANTS[B21] is B21 in component IKr (per_millivolt).
 * CONSTANTS[q21] is q21 in component IKr (dimensionless).
 * CONSTANTS[A31] is A31 in component IKr (per_millisecond).
 * CONSTANTS[B31] is B31 in component IKr (per_millivolt).
 * CONSTANTS[q31] is q31 in component IKr (dimensionless).
 * CONSTANTS[A41] is A41 in component IKr (per_millisecond).
 * CONSTANTS[B41] is B41 in component IKr (per_millivolt).
 * CONSTANTS[q41] is q41 in component IKr (dimensionless).
 * CONSTANTS[A51] is A51 in component IKr (per_millisecond).
 * CONSTANTS[B51] is B51 in component IKr (per_millivolt).
 * CONSTANTS[q51] is q51 in component IKr (dimensionless).
 * CONSTANTS[A52] is A52 in component IKr (per_millisecond).
 * CONSTANTS[B52] is B52 in component IKr (per_millivolt).
 * CONSTANTS[q52] is q52 in component IKr (dimensionless).
 * CONSTANTS[A53] is A53 in component IKr (per_millisecond).
 * CONSTANTS[B53] is B53 in component IKr (per_millivolt).
 * CONSTANTS[q53] is q53 in component IKr (dimensionless).
 * CONSTANTS[A61] is A61 in component IKr (per_millisecond).
 * CONSTANTS[B61] is B61 in component IKr (per_millivolt).
 * CONSTANTS[q61] is q61 in component IKr (dimensionless).
 * CONSTANTS[A62] is A62 in component IKr (per_millisecond).
 * CONSTANTS[B62] is B62 in component IKr (per_millivolt).
 * CONSTANTS[q62] is q62 in component IKr (dimensionless).
 * CONSTANTS[A63] is A63 in component IKr (per_millisecond).
 * CONSTANTS[B63] is B63 in component IKr (per_millivolt).
 * CONSTANTS[q63] is q63 in component IKr (dimensionless).
 * CONSTANTS[Kmax] is Kmax in component IKr (dimensionless).
 * CONSTANTS[Ku] is Ku in component IKr (per_millisecond).
 * CONSTANTS[n] is n in component IKr (dimensionless).
 * CONSTANTS[halfmax] is halfmax in component IKr (dimensionless).
 * CONSTANTS[Kt] is Kt in component IKr (per_millisecond).
 * CONSTANTS[Vhalf] is Vhalf in component IKr (millivolt).
 * CONSTANTS[Temp] is Temp in component IKr (dimensionless).
 * CONSTANTS[GKs_b] is GKs_b in component IKs (milliS_per_microF).
 * CONSTANTS[GKs] is GKs in component IKs (milliS_per_microF).
 * ALGEBRAIC[xs1ss] is xs1ss in component IKs (dimensionless).
 * ALGEBRAIC[xs2ss] is xs2ss in component IKs (dimensionless).
 * ALGEBRAIC[txs1] is txs1 in component IKs (millisecond).
 * CONSTANTS[txs1_max] is txs1_max in component IKs (millisecond).
 * STATES[xs1] is xs1 in component IKs (dimensionless).
 * STATES[xs2] is xs2 in component IKs (dimensionless).
 * ALGEBRAIC[KsCa] is KsCa in component IKs (dimensionless).
 * ALGEBRAIC[txs2] is txs2 in component IKs (millisecond).
 * CONSTANTS[GK1] is GK1 in component IK1 (milliS_per_microF).
 * CONSTANTS[GK1_b] is GK1_b in component IK1 (milliS_per_microF).
 * ALGEBRAIC[xk1ss] is xk1ss in component IK1 (dimensionless).
 * ALGEBRAIC[txk1] is txk1 in component IK1 (millisecond).
 * STATES[xk1] is xk1 in component IK1 (dimensionless).
 * ALGEBRAIC[rk1] is rk1 in component IK1 (millisecond).
 * CONSTANTS[kna1] is kna1 in component INaCa_i (per_millisecond).
 * CONSTANTS[kna2] is kna2 in component INaCa_i (per_millisecond).
 * CONSTANTS[kna3] is kna3 in component INaCa_i (per_millisecond).
 * CONSTANTS[kasymm] is kasymm in component INaCa_i (dimensionless).
 * CONSTANTS[wna] is wna in component INaCa_i (dimensionless).
 * CONSTANTS[wca] is wca in component INaCa_i (dimensionless).
 * CONSTANTS[wnaca] is wnaca in component INaCa_i (dimensionless).
 * CONSTANTS[kcaon] is kcaon in component INaCa_i (per_millisecond).
 * CONSTANTS[kcaoff] is kcaoff in component INaCa_i (per_millisecond).
 * CONSTANTS[qna] is qna in component INaCa_i (dimensionless).
 * CONSTANTS[qca] is qca in component INaCa_i (dimensionless).
 * ALGEBRAIC[hna] is hna in component INaCa_i (dimensionless).
 * ALGEBRAIC[hca] is hca in component INaCa_i (dimensionless).
 * CONSTANTS[KmCaAct] is KmCaAct in component INaCa_i (millimolar).
 * CONSTANTS[Gncx_b] is Gncx_b in component INaCa_i (milliS_per_microF).
 * CONSTANTS[Gncx] is Gncx in component INaCa_i (milliS_per_microF).
 * ALGEBRAIC[h1_i] is h1_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[h2_i] is h2_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[h3_i] is h3_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[h4_i] is h4_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[h5_i] is h5_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[h6_i] is h6_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[h7_i] is h7_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[h8_i] is h8_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[h9_i] is h9_i in component INaCa_i (dimensionless).
 * CONSTANTS[h10_i] is h10_i in component INaCa_i (dimensionless).
 * CONSTANTS[h11_i] is h11_i in component INaCa_i (dimensionless).
 * CONSTANTS[h12_i] is h12_i in component INaCa_i (dimensionless).
 * CONSTANTS[k1_i] is k1_i in component INaCa_i (dimensionless).
 * CONSTANTS[k2_i] is k2_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[k3p_i] is k3p_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[k3pp_i] is k3pp_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[k3_i] is k3_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[k4_i] is k4_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[k4p_i] is k4p_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[k4pp_i] is k4pp_i in component INaCa_i (dimensionless).
 * CONSTANTS[k5_i] is k5_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[k6_i] is k6_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[k7_i] is k7_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[k8_i] is k8_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[x1_i] is x1_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[x2_i] is x2_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[x3_i] is x3_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[x4_i] is x4_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[E1_i] is E1_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[E2_i] is E2_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[E3_i] is E3_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[E4_i] is E4_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[allo_i] is allo_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[JncxNa_i] is JncxNa_i in component INaCa_i (millimolar_per_millisecond).
 * ALGEBRAIC[JncxCa_i] is JncxCa_i in component INaCa_i (millimolar_per_millisecond).
 * ALGEBRAIC[h1_ss] is h1_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[h2_ss] is h2_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[h3_ss] is h3_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[h4_ss] is h4_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[h5_ss] is h5_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[h6_ss] is h6_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[h7_ss] is h7_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[h8_ss] is h8_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[h9_ss] is h9_ss in component INaCa_i (dimensionless).
 * CONSTANTS[h10_ss] is h10_ss in component INaCa_i (dimensionless).
 * CONSTANTS[h11_ss] is h11_ss in component INaCa_i (dimensionless).
 * CONSTANTS[h12_ss] is h12_ss in component INaCa_i (dimensionless).
 * CONSTANTS[k1_ss] is k1_ss in component INaCa_i (dimensionless).
 * CONSTANTS[k2_ss] is k2_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[k3p_ss] is k3p_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[k3pp_ss] is k3pp_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[k3_ss] is k3_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[k4_ss] is k4_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[k4p_ss] is k4p_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[k4pp_ss] is k4pp_ss in component INaCa_i (dimensionless).
 * CONSTANTS[k5_ss] is k5_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[k6_ss] is k6_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[k7_ss] is k7_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[k8_ss] is k8_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[x1_ss] is x1_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[x2_ss] is x2_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[x3_ss] is x3_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[x4_ss] is x4_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[E1_ss] is E1_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[E2_ss] is E2_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[E3_ss] is E3_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[E4_ss] is E4_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[allo_ss] is allo_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[JncxNa_ss] is JncxNa_ss in component INaCa_i (millimolar_per_millisecond).
 * ALGEBRAIC[JncxCa_ss] is JncxCa_ss in component INaCa_i (millimolar_per_millisecond).
 * CONSTANTS[k1p] is k1p in component INaK (per_millisecond).
 * CONSTANTS[k1m] is k1m in component INaK (per_millisecond).
 * CONSTANTS[k2p] is k2p in component INaK (per_millisecond).
 * CONSTANTS[k2m] is k2m in component INaK (per_millisecond).
 * CONSTANTS[k3p] is k3p in component INaK (per_millisecond).
 * CONSTANTS[k3m] is k3m in component INaK (per_millisecond).
 * CONSTANTS[k4p] is k4p in component INaK (per_millisecond).
 * CONSTANTS[k4m] is k4m in component INaK (per_millisecond).
 * CONSTANTS[Knai0] is Knai0 in component INaK (millimolar).
 * CONSTANTS[Knao0] is Knao0 in component INaK (millimolar).
 * CONSTANTS[delta] is delta in component INaK (millivolt).
 * CONSTANTS[Kki] is Kki in component INaK (per_millisecond).
 * CONSTANTS[Kko] is Kko in component INaK (per_millisecond).
 * CONSTANTS[MgADP] is MgADP in component INaK (millimolar).
 * CONSTANTS[MgATP] is MgATP in component INaK (millimolar).
 * CONSTANTS[Kmgatp] is Kmgatp in component INaK (millimolar).
 * CONSTANTS[H] is H in component INaK (millimolar).
 * CONSTANTS[eP] is eP in component INaK (dimensionless).
 * CONSTANTS[Khp] is Khp in component INaK (millimolar).
 * CONSTANTS[Knap] is Knap in component INaK (millimolar).
 * CONSTANTS[Kxkur] is Kxkur in component INaK (millimolar).
 * CONSTANTS[Pnak_b] is Pnak_b in component INaK (milliS_per_microF).
 * CONSTANTS[Pnak] is Pnak in component INaK (milliS_per_microF).
 * ALGEBRAIC[Knai] is Knai in component INaK (millimolar).
 * ALGEBRAIC[Knao] is Knao in component INaK (millimolar).
 * ALGEBRAIC[P] is P in component INaK (dimensionless).
 * ALGEBRAIC[a1] is a1 in component INaK (dimensionless).
 * CONSTANTS[b1] is b1 in component INaK (dimensionless).
 * CONSTANTS[a2] is a2 in component INaK (dimensionless).
 * ALGEBRAIC[b2] is b2 in component INaK (dimensionless).
 * ALGEBRAIC[a3] is a3 in component INaK (dimensionless).
 * ALGEBRAIC[b3] is b3 in component INaK (dimensionless).
 * CONSTANTS[a4] is a4 in component INaK (dimensionless).
 * ALGEBRAIC[b4] is b4 in component INaK (dimensionless).
 * ALGEBRAIC[x1] is x1 in component INaK (dimensionless).
 * ALGEBRAIC[x2] is x2 in component INaK (dimensionless).
 * ALGEBRAIC[x3] is x3 in component INaK (dimensionless).
 * ALGEBRAIC[x4] is x4 in component INaK (dimensionless).
 * ALGEBRAIC[E1] is E1 in component INaK (dimensionless).
 * ALGEBRAIC[E2] is E2 in component INaK (dimensionless).
 * ALGEBRAIC[E3] is E3 in component INaK (dimensionless).
 * ALGEBRAIC[E4] is E4 in component INaK (dimensionless).
 * ALGEBRAIC[JnakNa] is JnakNa in component INaK (millimolar_per_millisecond).
 * ALGEBRAIC[JnakK] is JnakK in component INaK (millimolar_per_millisecond).
 * ALGEBRAIC[xkb] is xkb in component IKb (dimensionless).
 * CONSTANTS[GKb_b] is GKb_b in component IKb (milliS_per_microF).
 * CONSTANTS[GKb] is GKb in component IKb (milliS_per_microF).
 * CONSTANTS[PNab] is PNab in component INab (milliS_per_microF).
 * ALGEBRAIC[A_Nab] is A in component INab (microA_per_microF).
 * CONSTANTS[B_Nab] is B in component INab (per_millivolt).
 * CONSTANTS[v0_Nab] is v0 in component INab (millivolt).
 * ALGEBRAIC[U] is U in component INab (dimensionless).
 * CONSTANTS[PCab] is PCab in component ICab (milliS_per_microF).
 * ALGEBRAIC[A_Cab] is A in component ICab (microA_per_microF).
 * CONSTANTS[B_Cab] is B in component ICab (per_millivolt).
 * CONSTANTS[v0_Cab] is v0 in component ICab (millivolt).
 * ALGEBRAIC[U] is U in component ICab (dimensionless).
 * CONSTANTS[GpCa] is GpCa in component IpCa (milliS_per_microF).
 * CONSTANTS[KmCap] is KmCap in component IpCa (millimolar).
 * CONSTANTS[bt] is bt in component ryr (millisecond).
 * CONSTANTS[a_rel] is a_rel in component ryr (millisecond).
 * ALGEBRAIC[Jrel_inf] is Jrel_inf in component ryr (dimensionless).
 * ALGEBRAIC[tau_rel] is tau_rel in component ryr (millisecond).
 * ALGEBRAIC[Jrel_infp] is Jrel_infp in component ryr (dimensionless).
 * ALGEBRAIC[Jrel_temp] is Jrel_temp in component ryr (dimensionless).
 * ALGEBRAIC[tau_relp] is tau_relp in component ryr (millisecond).
 * STATES[Jrelnp] is Jrelnp in component ryr (dimensionless).
 * STATES[Jrelp] is Jrelp in component ryr (dimensionless).
 * CONSTANTS[btp] is btp in component ryr (millisecond).
 * CONSTANTS[a_relp] is a_relp in component ryr (millisecond).
 * ALGEBRAIC[Jrel_inf_temp] is Jrel_inf_temp in component ryr (dimensionless).
 * ALGEBRAIC[fJrelp] is fJrelp in component ryr (dimensionless).
 * CONSTANTS[Jrel_scaling_factor] is Jrel_scaling_factor in component ryr (dimensionless).
 * ALGEBRAIC[tau_rel_temp] is tau_rel_temp in component ryr (millisecond).
 * ALGEBRAIC[tau_relp_temp] is tau_relp_temp in component ryr (millisecond).
 * CONSTANTS[upScale] is upScale in component SERCA (dimensionless).
 * ALGEBRAIC[Jupnp] is Jupnp in component SERCA (millimolar_per_millisecond).
 * ALGEBRAIC[Jupp] is Jupp in component SERCA (millimolar_per_millisecond).
 * ALGEBRAIC[fJupp] is fJupp in component SERCA (dimensionless).
 * ALGEBRAIC[Jleak] is Jleak in component SERCA (millimolar_per_millisecond).
 * CONSTANTS[Jup_b] is Jup_b in component SERCA (dimensionless).
 * RATES[V] is d/dt v in component membrane (millivolt).
 * RATES[CaMKt] is d/dt CaMKt in component CaMK (millimolar).
 * RATES[nai] is d/dt nai in component intracellular_ions (millimolar).
 * RATES[nass] is d/dt nass in component intracellular_ions (millimolar).
 * RATES[ki] is d/dt ki in component intracellular_ions (millimolar).
 * RATES[kss] is d/dt kss in component intracellular_ions (millimolar).
 * RATES[cai] is d/dt cai in component intracellular_ions (millimolar).
 * RATES[cass] is d/dt cass in component intracellular_ions (millimolar).
 * RATES[cansr] is d/dt cansr in component intracellular_ions (millimolar).
 * RATES[cajsr] is d/dt cajsr in component intracellular_ions (millimolar).
 * RATES[m] is d/dt m in component INa (dimensionless).
 * RATES[hf] is d/dt hf in component INa (dimensionless).
 * RATES[hs] is d/dt hs in component INa (dimensionless).
 * RATES[j] is d/dt j in component INa (dimensionless).
 * RATES[hsp] is d/dt hsp in component INa (dimensionless).
 * RATES[jp] is d/dt jp in component INa (dimensionless).
 * RATES[mL] is d/dt mL in component INaL (dimensionless).
 * RATES[hL] is d/dt hL in component INaL (dimensionless).
 * RATES[hLp] is d/dt hLp in component INaL (dimensionless).
 * RATES[a] is d/dt a in component Ito (dimensionless).
 * RATES[iF] is d/dt iF in component Ito (dimensionless).
 * RATES[iS] is d/dt iS in component Ito (dimensionless).
 * RATES[ap] is d/dt ap in component Ito (dimensionless).
 * RATES[iFp] is d/dt iFp in component Ito (dimensionless).
 * RATES[iSp] is d/dt iSp in component Ito (dimensionless).
 * RATES[d] is d/dt d in component ICaL (dimensionless).
 * RATES[ff] is d/dt ff in component ICaL (dimensionless).
 * RATES[fs] is d/dt fs in component ICaL (dimensionless).
 * RATES[fcaf] is d/dt fcaf in component ICaL (dimensionless).
 * RATES[fcas] is d/dt fcas in component ICaL (dimensionless).
 * RATES[jca] is d/dt jca in component ICaL (dimensionless).
 * RATES[ffp] is d/dt ffp in component ICaL (dimensionless).
 * RATES[fcafp] is d/dt fcafp in component ICaL (dimensionless).
 * RATES[nca] is d/dt nca in component ICaL (dimensionless).
 * RATES[IC1] is d/dt IC1 in component IKr (dimensionless).
 * RATES[IC2] is d/dt IC2 in component IKr (dimensionless).
 * RATES[C1] is d/dt C1 in component IKr (dimensionless).
 * RATES[C2] is d/dt C2 in component IKr (dimensionless).
 * RATES[O] is d/dt O in component IKr (dimensionless).
 * RATES[IO] is d/dt IO in component IKr (dimensionless).
 * RATES[IObound] is d/dt IObound in component IKr (dimensionless).
 * RATES[Obound] is d/dt Obound in component IKr (dimensionless).
 * RATES[Cbound] is d/dt Cbound in component IKr (dimensionless).
 * RATES[D] is d/dt D in component IKr (dimensionless).
 * RATES[xs1] is d/dt xs1 in component IKs (dimensionless).
 * RATES[xs2] is d/dt xs2 in component IKs (dimensionless).
 * RATES[xk1] is d/dt xk1 in component IK1 (dimensionless).
 * RATES[Jrelnp] is d/dt Jrelnp in component ryr (dimensionless).
 * RATES[Jrelp] is d/dt Jrelp in component ryr (dimensionless).
 */

// short ORd_num_of_algebraic = 200;
// short ORd_num_of_constants = 206;
// short ORd_num_of_states = 50;
// short ORd_num_of_rates = 50;

__device__ void ___initConsts(double *CONSTANTS, double *STATES, double type, double bcl, int offset)
{
CONSTANTS[(ORd_num_of_constants * offset) + celltype] = type;
CONSTANTS[(ORd_num_of_constants * offset) + nao] = 140;
CONSTANTS[(ORd_num_of_constants * offset) + cao] = 1.8;
CONSTANTS[(ORd_num_of_constants * offset) + ko] = 5.4;
CONSTANTS[(ORd_num_of_constants * offset) + R] = 8314;
CONSTANTS[(ORd_num_of_constants * offset) + T] = 310;
CONSTANTS[(ORd_num_of_constants * offset) + F] = 96485;
CONSTANTS[(ORd_num_of_constants * offset) + zna] = 1;
CONSTANTS[(ORd_num_of_constants * offset) + zca] = 2;
CONSTANTS[(ORd_num_of_constants * offset) + zk] = 1;
CONSTANTS[(ORd_num_of_constants * offset) + L] = 0.01;
CONSTANTS[(ORd_num_of_constants * offset) + rad] = 0.0011;
STATES[(ORd_num_of_states * offset) +  V] = -88.00190465;
CONSTANTS[(ORd_num_of_constants * offset) + stim_start] = 10;
CONSTANTS[(ORd_num_of_constants * offset) + stim_end] = 100000000000000000;
CONSTANTS[(ORd_num_of_constants * offset) + amp] = -80;
CONSTANTS[(ORd_num_of_constants * offset) + BCL] = 1000;
CONSTANTS[(ORd_num_of_constants * offset) + duration] = 0.5;
CONSTANTS[(ORd_num_of_constants * offset) + KmCaMK] = 0.15;
CONSTANTS[(ORd_num_of_constants * offset) + aCaMK] = 0.05;
CONSTANTS[(ORd_num_of_constants * offset) + bCaMK] = 0.00068;
CONSTANTS[(ORd_num_of_constants * offset) + CaMKo] = 0.05;
CONSTANTS[(ORd_num_of_constants * offset) + KmCaM] = 0.0015;
STATES[(ORd_num_of_states * offset) +  CaMKt] = 0.0125840447;
STATES[(ORd_num_of_states * offset) +  cass] = 8.49e-05;
CONSTANTS[(ORd_num_of_constants * offset) + cmdnmax_b] = 0.05;
CONSTANTS[(ORd_num_of_constants * offset) + kmcmdn] = 0.00238;
CONSTANTS[(ORd_num_of_constants * offset) + trpnmax] = 0.07;
CONSTANTS[(ORd_num_of_constants * offset) + kmtrpn] = 0.0005;
CONSTANTS[(ORd_num_of_constants * offset) + BSRmax] = 0.047;
CONSTANTS[(ORd_num_of_constants * offset) + KmBSR] = 0.00087;
CONSTANTS[(ORd_num_of_constants * offset) + BSLmax] = 1.124;
CONSTANTS[(ORd_num_of_constants * offset) + KmBSL] = 0.0087;
CONSTANTS[(ORd_num_of_constants * offset) + csqnmax] = 10;
CONSTANTS[(ORd_num_of_constants * offset) + kmcsqn] = 0.8;
STATES[(ORd_num_of_states * offset) +  nai] = 7.268004498;
STATES[(ORd_num_of_states * offset) +  nass] = 7.268089977;
STATES[(ORd_num_of_states * offset) +  ki] = 144.6555918;
STATES[(ORd_num_of_states * offset) +  kss] = 144.6555651;
STATES[(ORd_num_of_states * offset) +  cansr] = 1.619574538;
STATES[(ORd_num_of_states * offset) +  cajsr] = 1.571234014;
STATES[(ORd_num_of_states * offset) +  cai] = 8.6e-05;
CONSTANTS[(ORd_num_of_constants * offset) + cm] = 1;
CONSTANTS[(ORd_num_of_constants * offset) + PKNa] = 0.01833;
CONSTANTS[(ORd_num_of_constants * offset) + mssV1] = 39.57;
CONSTANTS[(ORd_num_of_constants * offset) + mssV2] = 9.871;
CONSTANTS[(ORd_num_of_constants * offset) + mtV1] = 11.64;
CONSTANTS[(ORd_num_of_constants * offset) + mtV2] = 34.77;
CONSTANTS[(ORd_num_of_constants * offset) + mtD1] = 6.765;
CONSTANTS[(ORd_num_of_constants * offset) + mtD2] = 8.552;
CONSTANTS[(ORd_num_of_constants * offset) + mtV3] = 77.42;
CONSTANTS[(ORd_num_of_constants * offset) + mtV4] = 5.955;
STATES[(ORd_num_of_states * offset) +  m] = 0.007344121102;
CONSTANTS[(ORd_num_of_constants * offset) + hssV1] = 82.9;
CONSTANTS[(ORd_num_of_constants * offset) + hssV2] = 6.086;
CONSTANTS[(ORd_num_of_constants * offset) + Ahf] = 0.99;
STATES[(ORd_num_of_states * offset) +  hf] = 0.6981071913;
STATES[(ORd_num_of_states * offset) +  hs] = 0.6980895801;
CONSTANTS[(ORd_num_of_constants * offset) + GNa] = 75;
CONSTANTS[(ORd_num_of_constants * offset) + shift_INa_inact] = 0;
STATES[(ORd_num_of_states * offset) +  j] = 0.6979908432;
STATES[(ORd_num_of_states * offset) +  hsp] = 0.4549485525;
STATES[(ORd_num_of_states * offset) +  jp] = 0.6979245865;
STATES[(ORd_num_of_states * offset) +  mL] = 0.0001882617273;
CONSTANTS[(ORd_num_of_constants * offset) + thL] = 200;
STATES[(ORd_num_of_states * offset) +  hL] = 0.5008548855;
STATES[(ORd_num_of_states * offset) + hLp] = 0.2693065357;
CONSTANTS[(ORd_num_of_constants * offset) + GNaL_b] = 0.019957499999999975;
CONSTANTS[(ORd_num_of_constants * offset) + Gto_b] = 0.02;
STATES[(ORd_num_of_states * offset) + a] = 0.001001097687;
STATES[(ORd_num_of_states * offset) + iF] = 0.9995541745;
STATES[(ORd_num_of_states * offset) + iS] = 0.5865061736;
STATES[(ORd_num_of_states * offset) + ap] = 0.0005100862934;
STATES[(ORd_num_of_states * offset) + iFp] = 0.9995541823;
STATES[(ORd_num_of_states * offset) + iSp] = 0.6393399482;
CONSTANTS[(ORd_num_of_constants * offset) + Kmn] = 0.002;
CONSTANTS[(ORd_num_of_constants * offset) + k2n] = 1000;
CONSTANTS[(ORd_num_of_constants * offset) + PCa_b] = 0.0001007;
STATES[(ORd_num_of_states * offset) + d] = 2.34e-9;
STATES[(ORd_num_of_states * offset) + ff] = 0.9999999909;
STATES[(ORd_num_of_states * offset) + fs] = 0.9102412777;
STATES[(ORd_num_of_states * offset) + fcaf] = 0.9999999909;
STATES[(ORd_num_of_states * offset) + fcas] = 0.9998046777;
STATES[(ORd_num_of_states * offset) + jca] = 0.9999738312;
STATES[(ORd_num_of_states * offset) + ffp] = 0.9999999909;
STATES[(ORd_num_of_states * offset) + fcafp] = 0.9999999909;
STATES[(ORd_num_of_states * offset) + nca] = 0.002749414044;
CONSTANTS[(ORd_num_of_constants * offset) + GKr_b] = 0.04658545454545456;
STATES[(ORd_num_of_states * offset) + IC1] = 0.999637;
STATES[(ORd_num_of_states * offset) + IC2] = 6.83208e-05;
STATES[(ORd_num_of_states * offset) + C1] = 1.80145e-08;
STATES[(ORd_num_of_states * offset) + C2] = 8.26619e-05;
STATES[(ORd_num_of_states * offset) + O] = 0.00015551;
STATES[(ORd_num_of_states * offset) + IO] = 5.67623e-05;
STATES[(ORd_num_of_states * offset) + IObound] = 0;
STATES[(ORd_num_of_states * offset) + Obound] = 0;
STATES[(ORd_num_of_states * offset) + Cbound] = 0;
STATES[(ORd_num_of_states * offset) + D] = 0;
CONSTANTS[(ORd_num_of_constants * offset) + A1] = 0.0264;
CONSTANTS[(ORd_num_of_constants * offset) + B1] = 4.631E-05;
CONSTANTS[(ORd_num_of_constants * offset) + q1] = 4.843;
CONSTANTS[(ORd_num_of_constants * offset) + A2] = 4.986E-06;
CONSTANTS[(ORd_num_of_constants * offset) + B2] = -0.004226;
CONSTANTS[(ORd_num_of_constants * offset) + q2] = 4.23;
CONSTANTS[(ORd_num_of_constants * offset) + A3] = 0.001214;
CONSTANTS[(ORd_num_of_constants * offset) + B3] = 0.008516;
CONSTANTS[(ORd_num_of_constants * offset) + q3] = 4.962;
CONSTANTS[(ORd_num_of_constants * offset) + A4] = 1.854E-05;
CONSTANTS[(ORd_num_of_constants * offset) + B4] = -0.04641;
CONSTANTS[(ORd_num_of_constants * offset) + q4] = 3.769;
CONSTANTS[(ORd_num_of_constants * offset) + A11] = 0.0007868;
CONSTANTS[(ORd_num_of_constants * offset) + B11] = 1.535E-08;
CONSTANTS[(ORd_num_of_constants * offset) + q11] = 4.942;
CONSTANTS[(ORd_num_of_constants * offset) + A21] = 5.455E-06;
CONSTANTS[(ORd_num_of_constants * offset) + B21] = -0.1688;
CONSTANTS[(ORd_num_of_constants * offset) + q21] = 4.156;
CONSTANTS[(ORd_num_of_constants * offset) + A31] = 0.005509;
CONSTANTS[(ORd_num_of_constants * offset) + B31] = 7.771E-09;
CONSTANTS[(ORd_num_of_constants * offset) + q31] = 4.22;
CONSTANTS[(ORd_num_of_constants * offset) + A41] = 0.001416;
CONSTANTS[(ORd_num_of_constants * offset) + B41] = -0.02877;
CONSTANTS[(ORd_num_of_constants * offset) + q41] = 1.459;
CONSTANTS[(ORd_num_of_constants * offset) + A51] = 0.4492;
CONSTANTS[(ORd_num_of_constants * offset) + B51] = 0.008595;
CONSTANTS[(ORd_num_of_constants * offset) + q51] = 5;
CONSTANTS[(ORd_num_of_constants * offset) + A52] = 0.3181;
CONSTANTS[(ORd_num_of_constants * offset) + B52] = 3.613E-08;
CONSTANTS[(ORd_num_of_constants * offset) + q52] = 4.663;
CONSTANTS[(ORd_num_of_constants * offset) + A53] = 0.149;
CONSTANTS[(ORd_num_of_constants * offset) + B53] = 0.004668;
CONSTANTS[(ORd_num_of_constants * offset) + q53] = 2.412;
CONSTANTS[(ORd_num_of_constants * offset) + A61] = 0.01241;
CONSTANTS[(ORd_num_of_constants * offset) + B61] = 0.1725;
CONSTANTS[(ORd_num_of_constants * offset) + q61] = 5.568;
CONSTANTS[(ORd_num_of_constants * offset) + A62] = 0.3226;
CONSTANTS[(ORd_num_of_constants * offset) + B62] = -0.0006575;
CONSTANTS[(ORd_num_of_constants * offset) + q62] = 5;
CONSTANTS[(ORd_num_of_constants * offset) + A63] = 0.008978;
CONSTANTS[(ORd_num_of_constants * offset) + B63] = -0.02215;
CONSTANTS[(ORd_num_of_constants * offset) + q63] = 5.682;
CONSTANTS[(ORd_num_of_constants * offset) + Kmax] = 0;
CONSTANTS[(ORd_num_of_constants * offset) + Ku] = 0;
CONSTANTS[(ORd_num_of_constants * offset) + n] = 1;
CONSTANTS[(ORd_num_of_constants * offset) + halfmax] = 1;
CONSTANTS[(ORd_num_of_constants * offset) + Kt] = 0;
CONSTANTS[(ORd_num_of_constants * offset) + Vhalf] = 1;
CONSTANTS[(ORd_num_of_constants * offset) + Temp] = 37;
CONSTANTS[(ORd_num_of_constants * offset) + GKs_b] = 0.006358000000000001;
CONSTANTS[(ORd_num_of_constants * offset) + txs1_max] = 817.3;
STATES[(ORd_num_of_states * offset) + xs1] = 0.2707758025;
STATES[(ORd_num_of_states * offset) + xs2] = 0.0001928503426;
CONSTANTS[(ORd_num_of_constants * offset) + GK1_b] = 0.3239783999999998;
STATES[(ORd_num_of_states * offset) + xk1] = 0.9967597594;
CONSTANTS[(ORd_num_of_constants * offset) + kna1] = 15;
CONSTANTS[(ORd_num_of_constants * offset) + kna2] = 5;
CONSTANTS[(ORd_num_of_constants * offset) + kna3] = 88.12;
CONSTANTS[(ORd_num_of_constants * offset) + kasymm] = 12.5;
CONSTANTS[(ORd_num_of_constants * offset) + wna] = 6e4;
CONSTANTS[(ORd_num_of_constants * offset) + wca] = 6e4;
CONSTANTS[(ORd_num_of_constants * offset) + wnaca] = 5e3;
CONSTANTS[(ORd_num_of_constants * offset) + kcaon] = 1.5e6;
CONSTANTS[(ORd_num_of_constants * offset) + kcaoff] = 5e3;
CONSTANTS[(ORd_num_of_constants * offset) + qna] = 0.5224;
CONSTANTS[(ORd_num_of_constants * offset) + qca] = 0.167;
CONSTANTS[(ORd_num_of_constants * offset) + KmCaAct] = 150e-6;
CONSTANTS[(ORd_num_of_constants * offset) + Gncx_b] = 0.0008;
CONSTANTS[(ORd_num_of_constants * offset) + k1p] = 949.5;
CONSTANTS[(ORd_num_of_constants * offset) + k1m] = 182.4;
CONSTANTS[(ORd_num_of_constants * offset) + k2p] = 687.2;
CONSTANTS[(ORd_num_of_constants * offset) + k2m] = 39.4;
CONSTANTS[(ORd_num_of_constants * offset) + k3p] = 1899;
CONSTANTS[(ORd_num_of_constants * offset) + k3m] = 79300;
CONSTANTS[(ORd_num_of_constants * offset) + k4p] = 639;
CONSTANTS[(ORd_num_of_constants * offset) + k4m] = 40;
CONSTANTS[(ORd_num_of_constants * offset) + Knai0] = 9.073;
CONSTANTS[(ORd_num_of_constants * offset) + Knao0] = 27.78;
CONSTANTS[(ORd_num_of_constants * offset) + delta] = -0.155;
CONSTANTS[(ORd_num_of_constants * offset) + Kki] = 0.5;
CONSTANTS[(ORd_num_of_constants * offset) + Kko] = 0.3582;
CONSTANTS[(ORd_num_of_constants * offset) + MgADP] = 0.05;
CONSTANTS[(ORd_num_of_constants * offset) + MgATP] = 9.8;
CONSTANTS[(ORd_num_of_constants * offset) + Kmgatp] = 1.698e-7;
CONSTANTS[(ORd_num_of_constants * offset) + H] = 1e-7;
CONSTANTS[(ORd_num_of_constants * offset) + eP] = 4.2;
CONSTANTS[(ORd_num_of_constants * offset) + Khp] = 1.698e-7;
CONSTANTS[(ORd_num_of_constants * offset) + Knap] = 224;
CONSTANTS[(ORd_num_of_constants * offset) + Kxkur] = 292;
CONSTANTS[(ORd_num_of_constants * offset) + Pnak_b] = 30;
CONSTANTS[(ORd_num_of_constants * offset) + GKb_b] = 0.003;
CONSTANTS[(ORd_num_of_constants * offset) + PNab] = 3.75e-10;
CONSTANTS[(ORd_num_of_constants * offset) + PCab] = 2.5e-8;
CONSTANTS[(ORd_num_of_constants * offset) + GpCa] = 0.0005;
CONSTANTS[(ORd_num_of_constants * offset) + KmCap] = 0.0005;
CONSTANTS[(ORd_num_of_constants * offset) + bt] = 4.75;
STATES[(ORd_num_of_states * offset) + Jrelnp] = 2.5e-7;
STATES[(ORd_num_of_states * offset) + Jrelp] = 3.12e-7;
CONSTANTS[(ORd_num_of_constants * offset) + Jrel_scaling_factor] = 1.0;
CONSTANTS[(ORd_num_of_constants * offset) + Jup_b] = 1.0;
CONSTANTS[(ORd_num_of_constants * offset) + frt] = CONSTANTS[(ORd_num_of_constants * offset) + F]/( CONSTANTS[(ORd_num_of_constants * offset) + R]*CONSTANTS[(ORd_num_of_constants * offset) + T]);
CONSTANTS[(ORd_num_of_constants * offset) + cmdnmax] = (CONSTANTS[(ORd_num_of_constants * offset) + celltype]==1.00000 ?  CONSTANTS[(ORd_num_of_constants * offset) + cmdnmax_b]*1.30000 : CONSTANTS[(ORd_num_of_constants * offset) + cmdnmax_b]);
CONSTANTS[(ORd_num_of_constants * offset) + Ahs] = 1.00000 - CONSTANTS[(ORd_num_of_constants * offset) + Ahf];
CONSTANTS[(ORd_num_of_constants * offset) + thLp] =  3.00000*CONSTANTS[(ORd_num_of_constants * offset) + thL];
CONSTANTS[(ORd_num_of_constants * offset) + GNaL] = (CONSTANTS[(ORd_num_of_constants * offset) + celltype]==1.00000 ?  CONSTANTS[(ORd_num_of_constants * offset) + GNaL_b]*0.600000 : CONSTANTS[(ORd_num_of_constants * offset) + GNaL_b]);
CONSTANTS[(ORd_num_of_constants * offset) + Gto] = (CONSTANTS[(ORd_num_of_constants * offset) + celltype]==1.00000 ?  CONSTANTS[(ORd_num_of_constants * offset) + Gto_b]*4.00000 : CONSTANTS[(ORd_num_of_constants * offset) + celltype]==2.00000 ?  CONSTANTS[(ORd_num_of_constants * offset) + Gto_b]*4.00000 : CONSTANTS[(ORd_num_of_constants * offset) + Gto_b]);
CONSTANTS[(ORd_num_of_constants * offset) + Aff] = 0.600000;
CONSTANTS[(ORd_num_of_constants * offset) + PCa] = (CONSTANTS[(ORd_num_of_constants * offset) + celltype]==1.00000 ?  CONSTANTS[(ORd_num_of_constants * offset) + PCa_b]*1.20000 : CONSTANTS[(ORd_num_of_constants * offset) + celltype]==2.00000 ?  CONSTANTS[(ORd_num_of_constants * offset) + PCa_b]*2.50000 : CONSTANTS[(ORd_num_of_constants * offset) + PCa_b]);
CONSTANTS[(ORd_num_of_constants * offset) + tjca] = 75.0000;
CONSTANTS[(ORd_num_of_constants * offset) + v0_CaL] = 0.000000;
CONSTANTS[(ORd_num_of_constants * offset) + GKr] = (CONSTANTS[(ORd_num_of_constants * offset) + celltype]==1.00000 ?  CONSTANTS[(ORd_num_of_constants * offset) + GKr_b]*1.30000 : CONSTANTS[(ORd_num_of_constants * offset) + celltype]==2.00000 ?  CONSTANTS[(ORd_num_of_constants * offset) + GKr_b]*0.800000 : CONSTANTS[(ORd_num_of_constants * offset) + GKr_b]);
CONSTANTS[(ORd_num_of_constants * offset) + GKs] = (CONSTANTS[(ORd_num_of_constants * offset) + celltype]==1.00000 ?  CONSTANTS[(ORd_num_of_constants * offset) + GKs_b]*1.40000 : CONSTANTS[(ORd_num_of_constants * offset) + GKs_b]);
CONSTANTS[(ORd_num_of_constants * offset) + GK1] = (CONSTANTS[(ORd_num_of_constants * offset) + celltype]==1.00000 ?  CONSTANTS[(ORd_num_of_constants * offset) + GK1_b]*1.20000 : CONSTANTS[(ORd_num_of_constants * offset) + celltype]==2.00000 ?  CONSTANTS[(ORd_num_of_constants * offset) + GK1_b]*1.30000 : CONSTANTS[(ORd_num_of_constants * offset) + GK1_b]);
CONSTANTS[(ORd_num_of_constants * offset) + vcell] =  1000.00*3.14000*CONSTANTS[(ORd_num_of_constants * offset) + rad]*CONSTANTS[(ORd_num_of_constants * offset) + rad]*CONSTANTS[(ORd_num_of_constants * offset) + L];
CONSTANTS[(ORd_num_of_constants * offset) + GKb] = (CONSTANTS[(ORd_num_of_constants * offset) + celltype]==1.00000 ?  CONSTANTS[(ORd_num_of_constants * offset) + GKb_b]*0.600000 : CONSTANTS[(ORd_num_of_constants * offset) + GKb_b]);
CONSTANTS[(ORd_num_of_constants * offset) + v0_Nab] = 0.000000;
CONSTANTS[(ORd_num_of_constants * offset) + v0_Cab] = 0.000000;
CONSTANTS[(ORd_num_of_constants * offset) + a_rel] =  0.500000*CONSTANTS[(ORd_num_of_constants * offset) + bt];
CONSTANTS[(ORd_num_of_constants * offset) + btp] =  1.25000*CONSTANTS[(ORd_num_of_constants * offset) + bt];
CONSTANTS[(ORd_num_of_constants * offset) + upScale] = (CONSTANTS[(ORd_num_of_constants * offset) + celltype]==1.00000 ? 1.30000 : 1.00000);
CONSTANTS[(ORd_num_of_constants * offset) + cnc] = 0.000000;
CONSTANTS[(ORd_num_of_constants * offset) + ffrt] =  CONSTANTS[(ORd_num_of_constants * offset) + F]*CONSTANTS[(ORd_num_of_constants * offset) + frt];
CONSTANTS[(ORd_num_of_constants * offset) + Afs] = 1.00000 - CONSTANTS[(ORd_num_of_constants * offset) + Aff];
CONSTANTS[(ORd_num_of_constants * offset) + PCap] =  1.10000*CONSTANTS[(ORd_num_of_constants * offset) + PCa];
CONSTANTS[(ORd_num_of_constants * offset) + PCaNa] =  0.00125000*CONSTANTS[(ORd_num_of_constants * offset) + PCa];
CONSTANTS[(ORd_num_of_constants * offset) + PCaK] =  0.000357400*CONSTANTS[(ORd_num_of_constants * offset) + PCa];
CONSTANTS[(ORd_num_of_constants * offset) + B_1] =  2.00000*CONSTANTS[(ORd_num_of_constants * offset) + frt];
CONSTANTS[(ORd_num_of_constants * offset) + B_2] = CONSTANTS[(ORd_num_of_constants * offset) + frt];
CONSTANTS[(ORd_num_of_constants * offset) + B_3] = CONSTANTS[(ORd_num_of_constants * offset) + frt];
CONSTANTS[(ORd_num_of_constants * offset) + Ageo] =  2.00000*3.14000*CONSTANTS[(ORd_num_of_constants * offset) + rad]*CONSTANTS[(ORd_num_of_constants * offset) + rad]+ 2.00000*3.14000*CONSTANTS[(ORd_num_of_constants * offset) + rad]*CONSTANTS[(ORd_num_of_constants * offset) + L];
CONSTANTS[(ORd_num_of_constants * offset) + B_Nab] = CONSTANTS[(ORd_num_of_constants * offset) + frt];
CONSTANTS[(ORd_num_of_constants * offset) + B_Cab] =  2.00000*CONSTANTS[(ORd_num_of_constants * offset) + frt];
CONSTANTS[(ORd_num_of_constants * offset) + a_relp] =  0.500000*CONSTANTS[(ORd_num_of_constants * offset) + btp];
CONSTANTS[(ORd_num_of_constants * offset) + PCaNap] =  0.00125000*CONSTANTS[(ORd_num_of_constants * offset) + PCap];
CONSTANTS[(ORd_num_of_constants * offset) + PCaKp] =  0.000357400*CONSTANTS[(ORd_num_of_constants * offset) + PCap];
CONSTANTS[(ORd_num_of_constants * offset) + Acap] =  2.00000*CONSTANTS[(ORd_num_of_constants * offset) + Ageo];
CONSTANTS[(ORd_num_of_constants * offset) + vmyo] =  0.680000*CONSTANTS[(ORd_num_of_constants * offset) + vcell];
CONSTANTS[(ORd_num_of_constants * offset) + vnsr] =  0.0552000*CONSTANTS[(ORd_num_of_constants * offset) + vcell];
CONSTANTS[(ORd_num_of_constants * offset) + vjsr] =  0.00480000*CONSTANTS[(ORd_num_of_constants * offset) + vcell];
CONSTANTS[(ORd_num_of_constants * offset) + vss] =  0.0200000*CONSTANTS[(ORd_num_of_constants * offset) + vcell];
CONSTANTS[(ORd_num_of_constants * offset) + h10_i] = CONSTANTS[(ORd_num_of_constants * offset) + kasymm]+1.00000+ (CONSTANTS[(ORd_num_of_constants * offset) + nao]/CONSTANTS[(ORd_num_of_constants * offset) + kna1])*(1.00000+CONSTANTS[(ORd_num_of_constants * offset) + nao]/CONSTANTS[(ORd_num_of_constants * offset) + kna2]);
CONSTANTS[(ORd_num_of_constants * offset) + h11_i] = ( CONSTANTS[(ORd_num_of_constants * offset) + nao]*CONSTANTS[(ORd_num_of_constants * offset) + nao])/( CONSTANTS[(ORd_num_of_constants * offset) + h10_i]*CONSTANTS[(ORd_num_of_constants * offset) + kna1]*CONSTANTS[(ORd_num_of_constants * offset) + kna2]);
CONSTANTS[(ORd_num_of_constants * offset) + h12_i] = 1.00000/CONSTANTS[(ORd_num_of_constants * offset) + h10_i];
CONSTANTS[(ORd_num_of_constants * offset) + k1_i] =  CONSTANTS[(ORd_num_of_constants * offset) + h12_i]*CONSTANTS[(ORd_num_of_constants * offset) + cao]*CONSTANTS[(ORd_num_of_constants * offset) + kcaon];
CONSTANTS[(ORd_num_of_constants * offset) + k2_i] = CONSTANTS[(ORd_num_of_constants * offset) + kcaoff];
CONSTANTS[(ORd_num_of_constants * offset) + k5_i] = CONSTANTS[(ORd_num_of_constants * offset) + kcaoff];
CONSTANTS[(ORd_num_of_constants * offset) + Gncx] = (CONSTANTS[(ORd_num_of_constants * offset) + celltype]==1.00000 ?  CONSTANTS[(ORd_num_of_constants * offset) + Gncx_b]*1.10000 : CONSTANTS[(ORd_num_of_constants * offset) + celltype]==2.00000 ?  CONSTANTS[(ORd_num_of_constants * offset) + Gncx_b]*1.40000 : CONSTANTS[(ORd_num_of_constants * offset) + Gncx_b]);
CONSTANTS[(ORd_num_of_constants * offset) + h10_ss] = CONSTANTS[(ORd_num_of_constants * offset) + kasymm]+1.00000+ (CONSTANTS[(ORd_num_of_constants * offset) + nao]/CONSTANTS[(ORd_num_of_constants * offset) + kna1])*(1.00000+CONSTANTS[(ORd_num_of_constants * offset) + nao]/CONSTANTS[(ORd_num_of_constants * offset) + kna2]);
CONSTANTS[(ORd_num_of_constants * offset) + h11_ss] = ( CONSTANTS[(ORd_num_of_constants * offset) + nao]*CONSTANTS[(ORd_num_of_constants * offset) + nao])/( CONSTANTS[(ORd_num_of_constants * offset) + h10_ss]*CONSTANTS[(ORd_num_of_constants * offset) + kna1]*CONSTANTS[(ORd_num_of_constants * offset) + kna2]);
CONSTANTS[(ORd_num_of_constants * offset) + h12_ss] = 1.00000/CONSTANTS[(ORd_num_of_constants * offset) + h10_ss];
CONSTANTS[(ORd_num_of_constants * offset) + k1_ss] =  CONSTANTS[(ORd_num_of_constants * offset) + h12_ss]*CONSTANTS[(ORd_num_of_constants * offset) + cao]*CONSTANTS[(ORd_num_of_constants * offset) + kcaon];
CONSTANTS[(ORd_num_of_constants * offset) + k2_ss] = CONSTANTS[(ORd_num_of_constants * offset) + kcaoff];
CONSTANTS[(ORd_num_of_constants * offset) + k5_ss] = CONSTANTS[(ORd_num_of_constants * offset) + kcaoff];
CONSTANTS[(ORd_num_of_constants * offset) + b1] =  CONSTANTS[(ORd_num_of_constants * offset) + k1m]*CONSTANTS[(ORd_num_of_constants * offset) + MgADP];
CONSTANTS[(ORd_num_of_constants * offset) + a2] = CONSTANTS[(ORd_num_of_constants * offset) + k2p];
CONSTANTS[(ORd_num_of_constants * offset) + a4] = (( CONSTANTS[(ORd_num_of_constants * offset) + k4p]*CONSTANTS[(ORd_num_of_constants * offset) + MgATP])/CONSTANTS[(ORd_num_of_constants * offset) + Kmgatp])/(1.00000+CONSTANTS[(ORd_num_of_constants * offset) + MgATP]/CONSTANTS[(ORd_num_of_constants * offset) + Kmgatp]);
CONSTANTS[(ORd_num_of_constants * offset) + Pnak] = (CONSTANTS[(ORd_num_of_constants * offset) + celltype]==1.00000 ?  CONSTANTS[(ORd_num_of_constants * offset) + Pnak_b]*0.900000 : CONSTANTS[(ORd_num_of_constants * offset) + celltype]==2.00000 ?  CONSTANTS[(ORd_num_of_constants * offset) + Pnak_b]*0.700000 : CONSTANTS[(ORd_num_of_constants * offset) + Pnak_b]);
}

__device__ void applyDrugEffect(double *CONSTANTS, double conc, double *ic50, double epsilon, int offset)
{
CONSTANTS[(ORd_num_of_constants * offset) + GK1] = CONSTANTS[(ORd_num_of_constants * offset) + GK1] * ((ic50[(offset*14) + 2] > 10E-14 && ic50[(offset*14) + 3] > 10E-14) ? 1./(1.+pow(conc/ic50[(offset*14) + 2],ic50[(offset*14) + 3])) : 1.);
CONSTANTS[(ORd_num_of_constants * offset) + GKs] = CONSTANTS[(ORd_num_of_constants * offset) + GKs] * ((ic50[(offset*14) + 4] > 10E-14 && ic50[(offset*14) + 5] > 10E-14) ? 1./(1.+pow(conc/ic50[(offset*14) + 4],ic50[(offset*14) + 5])) : 1.);
CONSTANTS[(ORd_num_of_constants * offset) + GNaL] = CONSTANTS[(ORd_num_of_constants * offset) + GNaL] * ((ic50[(offset*14) + 8] > 10E-14 && ic50[(offset*14) + 9] > 10E-14) ? 1./(1.+pow(conc/ic50[(offset*14) + 8],ic50[(offset*14) + 9])) : 1.);
CONSTANTS[(ORd_num_of_constants * offset) + GNa] = CONSTANTS[(ORd_num_of_constants * offset) + GNa] * ((ic50[(offset*14) + 6] > 10E-14 && ic50[(offset*14) + 7] > 10E-14) ? 1./(1.+pow(conc/ic50[(offset*14) + 6],ic50[(offset*14) + 7])) : 1.);
CONSTANTS[(ORd_num_of_constants * offset) + Gto] = CONSTANTS[(ORd_num_of_constants * offset) + Gto] * ((ic50[(offset*14) + 10] > 10E-14 && ic50[(offset*14) + 11] > 10E-14) ? 1./(1.+pow(conc/ic50[(offset*14) + 10],ic50[(offset*14) + 11])) : 1.);
CONSTANTS[(ORd_num_of_constants * offset) + PCa] = CONSTANTS[(ORd_num_of_constants * offset) + PCa] * ( (ic50[(offset*14) + 0] > 10E-14 && ic50[(offset*14) + 1] > 10E-14) ? 1./(1.+pow(conc/ic50[(offset*14) + 0],ic50[(offset*14) + 1])) : 1.);
}

__device__ void ___applyHERGBinding(double *CONSTANTS, double *STATES, double conc, double *herg, int offset)
{
if(conc > 10E-14){
CONSTANTS[(ORd_num_of_constants * offset) + Kmax] = herg[0];
CONSTANTS[(ORd_num_of_constants * offset) + Ku] = herg[1];
CONSTANTS[(ORd_num_of_constants * offset) + n] = herg[2];
CONSTANTS[(ORd_num_of_constants * offset) + halfmax] = herg[3];
CONSTANTS[(ORd_num_of_constants * offset) + Vhalf] = herg[4];
CONSTANTS[(ORd_num_of_constants * offset) + cnc] = conc;
STATES[(ORd_num_of_states * offset) + D] = CONSTANTS[(ORd_num_of_constants * offset) + cnc];
}
}

// void ohara_rudy_cipa_v1_2017::initConsts()
// {
// 	___initConsts(0.);
// }

// void ohara_rudy_cipa_v1_2017::initConsts(double type)
// {
// 	___initConsts(type);
// }

__device__ void initConsts(double *CONSTANTS, double *STATES, double type, double conc, double *ic50, double *herg, double *cvar, bool is_dutta, bool is_cvar, double bcl, double epsilon, int offset)
{
	___initConsts(CONSTANTS, STATES, type, bcl, offset);
	// mpi_printf(0,"Celltype: %lf\n", CONSTANTS[celltype]);
	// mpi_printf(0,"Control %lf %lf %lf %lf %lf\n", CONSTANTS[PCa], CONSTANTS[GK1], CONSTANTS[GKs], CONSTANTS[GNaL], CONSTANTS[GKr]);
	applyDrugEffect(CONSTANTS, conc, ic50, epsilon, offset);
	// mpi_printf(0,"After drug %lf %lf %lf %lf %lf\n", CONSTANTS[PCa], CONSTANTS[GK1], CONSTANTS[GKs], CONSTANTS[GNaL], CONSTANTS[GKr]);
	if (offset == 1) printf("in ord: Control hERG binding %lf %lf %lf %lf %lf %lf\n", CONSTANTS[(ORd_num_of_constants * offset) + Kmax], CONSTANTS[(ORd_num_of_constants * offset) + Ku], CONSTANTS[(ORd_num_of_constants * offset) + n], CONSTANTS[(ORd_num_of_constants * offset) + halfmax], CONSTANTS[ (ORd_num_of_constants * offset) + Vhalf], CONSTANTS[(ORd_num_of_constants * offset) + cnc]);
	___applyHERGBinding(CONSTANTS, STATES, conc, herg, offset);
	if (offset == 1) printf("in ord: Bootstrapped hERG binding %lf %lf %lf %lf %lf %lf\n", CONSTANTS[(ORd_num_of_constants * offset) + Kmax], CONSTANTS[(ORd_num_of_constants * offset) + Ku], CONSTANTS[(ORd_num_of_constants * offset) + n], CONSTANTS[(ORd_num_of_constants * offset) + halfmax], CONSTANTS[ (ORd_num_of_constants * offset) + Vhalf], CONSTANTS[(ORd_num_of_constants * offset) + cnc]);
}

__device__ void computeRates( double TIME, double *CONSTANTS, double *RATES, double *STATES, double *ALGEBRAIC, int offset, double land_trpn )
{
ALGEBRAIC[(ORd_num_of_algebraic * offset) + hLss] = 1.00000/(1.00000+exp((STATES[(ORd_num_of_states * offset) + V]+87.6100)/7.48800));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + hLssp] = 1.00000/(1.00000+exp((STATES[(ORd_num_of_states * offset) + V]+93.8100)/7.48800));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + mss] = 1.00000/(1.00000+exp(- (STATES[(ORd_num_of_states * offset) + V]+CONSTANTS[mssV1])/CONSTANTS[(ORd_num_of_constants * offset) + mssV2]));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + tm] = 1.00000/( CONSTANTS[(ORd_num_of_constants * offset) + mtD1]*exp((STATES[(ORd_num_of_states * offset) + V]+CONSTANTS[(ORd_num_of_constants * offset) + mtV1])/CONSTANTS[(ORd_num_of_constants * offset) + mtV2])+ CONSTANTS[(ORd_num_of_constants * offset) + mtD2]*exp(- (STATES[(ORd_num_of_states * offset) + V]+CONSTANTS[(ORd_num_of_constants * offset) + mtV3])/CONSTANTS[(ORd_num_of_constants * offset) + mtV4]));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + hss] = 1.00000/(1.00000+exp(((STATES[(ORd_num_of_states * offset) + V]+CONSTANTS[(ORd_num_of_constants * offset) + hssV1]) - CONSTANTS[(ORd_num_of_constants * offset) + shift_INa_inact])/CONSTANTS[(ORd_num_of_constants * offset) + hssV2]));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + thf] = 1.00000/( 1.43200e-05*exp(- ((STATES[(ORd_num_of_states * offset) + V]+1.19600) - CONSTANTS[(ORd_num_of_constants * offset) + shift_INa_inact])/6.28500)+ 6.14900*exp(((STATES[(ORd_num_of_states * offset) + V]+0.509600) - CONSTANTS[(ORd_num_of_constants * offset) + shift_INa_inact])/20.2700));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + ths] = 1.00000/( 0.00979400*exp(- ((STATES[(ORd_num_of_states * offset) + V]+17.9500) - CONSTANTS[(ORd_num_of_constants * offset) + shift_INa_inact])/28.0500)+ 0.334300*exp(((STATES[(ORd_num_of_states * offset) + V]+5.73000) - CONSTANTS[(ORd_num_of_constants * offset) + shift_INa_inact])/56.6600));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + ass] = 1.00000/(1.00000+exp(- (STATES[(ORd_num_of_states * offset) + V] - 14.3400)/14.8200));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + ta] = 1.05150/(1.00000/( 1.20890*(1.00000+exp(- (STATES[(ORd_num_of_states * offset) + V] - 18.4099)/29.3814)))+3.50000/(1.00000+exp((STATES[(ORd_num_of_states * offset) + V]+100.000)/29.3814)));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + dss] = 1.00000/(1.00000+exp(- (STATES[(ORd_num_of_states * offset) + V]+3.94000)/4.23000));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + td] = 0.600000+1.00000/(exp( - 0.0500000*(STATES[(ORd_num_of_states * offset) + V]+6.00000))+exp( 0.0900000*(STATES[(ORd_num_of_states * offset) + V]+14.0000)));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + fss] = 1.00000/(1.00000+exp((STATES[(ORd_num_of_states * offset) + V]+19.5800)/3.69600));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + tff] = 7.00000+1.00000/( 0.00450000*exp(- (STATES[(ORd_num_of_states * offset) + V]+20.0000)/10.0000)+ 0.00450000*exp((STATES[(ORd_num_of_states * offset) + V]+20.0000)/10.0000));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + tfs] = 1000.00+1.00000/( 3.50000e-05*exp(- (STATES[(ORd_num_of_states * offset) + V]+5.00000)/4.00000)+ 3.50000e-05*exp((STATES[(ORd_num_of_states * offset) + V]+5.00000)/6.00000));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + fcass] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + fss];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + km2n] =  STATES[(ORd_num_of_states * offset) + jca]*1.00000;
ALGEBRAIC[(ORd_num_of_algebraic * offset) + anca] = 1.00000/(CONSTANTS[(ORd_num_of_constants * offset) + k2n]/ALGEBRAIC[(ORd_num_of_algebraic * offset) + km2n]+pow(1.00000+CONSTANTS[(ORd_num_of_constants * offset) + Kmn]/STATES[(ORd_num_of_states * offset) + cass], 4.00000));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + xs1ss] = 1.00000/(1.00000+exp(- (STATES[(ORd_num_of_states * offset) + V]+11.6000)/8.93200));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + txs1] = CONSTANTS[(ORd_num_of_constants * offset) + txs1_max]+1.00000/( 0.000232600*exp((STATES[(ORd_num_of_states * offset) + V]+48.2800)/17.8000)+ 0.00129200*exp(- (STATES[(ORd_num_of_states * offset) + V]+210.000)/230.000));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + xk1ss] = 1.00000/(1.00000+exp(- (STATES[(ORd_num_of_states * offset) + V]+ 2.55380*CONSTANTS[(ORd_num_of_constants * offset) + ko]+144.590)/( 1.56920*CONSTANTS[(ORd_num_of_constants * offset) + ko]+3.81150)));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + txk1] = 122.200/(exp(- (STATES[(ORd_num_of_states * offset) + V]+127.200)/20.3600)+exp((STATES[(ORd_num_of_states * offset) + V]+236.800)/69.3300));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + CaMKb] = ( CONSTANTS[(ORd_num_of_constants * offset) + CaMKo]*(1.00000 - STATES[(ORd_num_of_states * offset) + CaMKt]))/(1.00000+CONSTANTS[(ORd_num_of_constants * offset) + KmCaM]/STATES[(ORd_num_of_states * offset) + cass]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + jss] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + hss];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + tj] = 2.03800+1.00000/( 0.0213600*exp(- ((STATES[(ORd_num_of_states * offset) + V]+100.600) - CONSTANTS[(ORd_num_of_constants * offset) + shift_INa_inact])/8.28100)+ 0.305200*exp(((STATES[(ORd_num_of_states * offset) + V]+0.994100) - CONSTANTS[(ORd_num_of_constants * offset) + shift_INa_inact])/38.4500));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + assp] = 1.00000/(1.00000+exp(- (STATES[(ORd_num_of_states * offset) + V] - 24.3400)/14.8200));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + tfcaf] = 7.00000+1.00000/( 0.0400000*exp(- (STATES[(ORd_num_of_states * offset) + V] - 4.00000)/7.00000)+ 0.0400000*exp((STATES[(ORd_num_of_states * offset) + V] - 4.00000)/7.00000));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + tfcas] = 100.000+1.00000/( 0.000120000*exp(- STATES[(ORd_num_of_states * offset) + V]/3.00000)+ 0.000120000*exp(STATES[(ORd_num_of_states * offset) + V]/7.00000));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + tffp] =  2.50000*ALGEBRAIC[(ORd_num_of_algebraic * offset) + tff];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + xs2ss] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + xs1ss];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + txs2] = 1.00000/( 0.0100000*exp((STATES[(ORd_num_of_states * offset) + V] - 50.0000)/20.0000)+ 0.0193000*exp(- (STATES[(ORd_num_of_states * offset) + V]+66.5400)/31.0000));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + hssp] = 1.00000/(1.00000+exp(((STATES[(ORd_num_of_states * offset) + V]+89.1000) - CONSTANTS[(ORd_num_of_constants * offset) + shift_INa_inact])/6.08600));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + thsp] =  3.00000*ALGEBRAIC[(ORd_num_of_algebraic * offset) + ths];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + tjp] =  1.46000*ALGEBRAIC[(ORd_num_of_algebraic * offset) + tj];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + mLss] = 1.00000/(1.00000+exp(- (STATES[(ORd_num_of_states * offset) + V]+42.8500)/5.26400));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + tmL] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + tm];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + tfcafp] =  2.50000*ALGEBRAIC[(ORd_num_of_algebraic * offset) + tfcaf];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + iss] = 1.00000/(1.00000+exp((STATES[(ORd_num_of_states * offset) + V]+43.9400)/5.71100));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + delta_epi] = (CONSTANTS[(ORd_num_of_constants * offset) + celltype]==1.00000 ? 1.00000 - 0.950000/(1.00000+exp((STATES[(ORd_num_of_states * offset) + V]+70.0000)/5.00000)) : 1.00000);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + tiF_b] = 4.56200+1.00000/( 0.393300*exp(- (STATES[(ORd_num_of_states * offset) + V]+100.000)/100.000)+ 0.0800400*exp((STATES[(ORd_num_of_states * offset) + V]+50.0000)/16.5900));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + tiF] =  ALGEBRAIC[(ORd_num_of_algebraic * offset) + tiF_b]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + delta_epi];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + tiS_b] = 23.6200+1.00000/( 0.00141600*exp(- (STATES[(ORd_num_of_states * offset) + V]+96.5200)/59.0500)+ 1.78000e-08*exp((STATES[(ORd_num_of_states * offset) + V]+114.100)/8.07900));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + tiS] =  ALGEBRAIC[(ORd_num_of_algebraic * offset) + tiS_b]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + delta_epi];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + dti_develop] = 1.35400+0.000100000/(exp((STATES[(ORd_num_of_states * offset) + V] - 167.400)/15.8900)+exp(- (STATES[(ORd_num_of_states * offset) + V] - 12.2300)/0.215400));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + dti_recover] = 1.00000 - 0.500000/(1.00000+exp((STATES[(ORd_num_of_states * offset) + V]+70.0000)/20.0000));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + tiFp] =  ALGEBRAIC[(ORd_num_of_algebraic * offset) + dti_develop]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + dti_recover]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + tiF];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + tiSp] =  ALGEBRAIC[(ORd_num_of_algebraic * offset) + dti_develop]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + dti_recover]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + tiS];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + f] =  CONSTANTS[(ORd_num_of_constants * offset) + Aff]*STATES[(ORd_num_of_states * offset) + ff]+ CONSTANTS[(ORd_num_of_constants * offset) + Afs]*STATES[(ORd_num_of_states * offset) + fs];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + Afcaf] = 0.300000+0.600000/(1.00000+exp((STATES[(ORd_num_of_states * offset) + V] - 10.0000)/10.0000));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + Afcas] = 1.00000 - ALGEBRAIC[(ORd_num_of_algebraic * offset) + Afcaf];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + fca] =  ALGEBRAIC[(ORd_num_of_algebraic * offset) + Afcaf]*STATES[(ORd_num_of_states * offset) + fcaf]+ ALGEBRAIC[(ORd_num_of_algebraic * offset) + Afcas]*STATES[(ORd_num_of_states * offset) + fcas];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + fp] =  CONSTANTS[(ORd_num_of_constants * offset) + Aff]*STATES[(ORd_num_of_states * offset) + ffp]+ CONSTANTS[(ORd_num_of_constants * offset) + Afs]*STATES[(ORd_num_of_states * offset) + fs];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + fcap] =  ALGEBRAIC[(ORd_num_of_algebraic * offset) + Afcaf]*STATES[(ORd_num_of_states * offset) + fcafp]+ ALGEBRAIC[(ORd_num_of_algebraic * offset) + Afcas]*STATES[(ORd_num_of_states * offset) + fcas];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + vfrt] =  STATES[(ORd_num_of_states * offset) + V]*CONSTANTS[(ORd_num_of_constants * offset) + frt];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + A_1] = ( 4.00000*CONSTANTS[(ORd_num_of_constants * offset) + ffrt]*( STATES[(ORd_num_of_states * offset) + cass]*exp( 2.00000*ALGEBRAIC[(ORd_num_of_algebraic * offset) + vfrt]) -  0.341000*CONSTANTS[(ORd_num_of_constants * offset) + cao]))/CONSTANTS[(ORd_num_of_constants * offset) + B_1];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + U_1] =  CONSTANTS[(ORd_num_of_constants * offset) + B_1]*(STATES[(ORd_num_of_states * offset) + V] - CONSTANTS[(ORd_num_of_constants * offset) + v0_CaL]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + PhiCaL] = (- 1.00000e-07<=ALGEBRAIC[(ORd_num_of_algebraic * offset) + U_1]&&ALGEBRAIC[(ORd_num_of_algebraic * offset) + U_1]<=1.00000e-07 ?  ALGEBRAIC[(ORd_num_of_algebraic * offset) + A_1]*(1.00000 -  0.500000*ALGEBRAIC[(ORd_num_of_algebraic * offset) + U_1]) : ( ALGEBRAIC[(ORd_num_of_algebraic * offset) + A_1]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + U_1])/(exp(ALGEBRAIC[(ORd_num_of_algebraic * offset) + U_1]) - 1.00000));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + CaMKa] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + CaMKb]+STATES[(ORd_num_of_states * offset) + CaMKt];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + fICaLp] = 1.00000/(1.00000+CONSTANTS[(ORd_num_of_constants * offset) + KmCaMK]/ALGEBRAIC[(ORd_num_of_algebraic * offset) + CaMKa]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + ICaL] =  (1.00000 - ALGEBRAIC[(ORd_num_of_algebraic * offset) + fICaLp])*CONSTANTS[(ORd_num_of_constants * offset) + PCa]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + PhiCaL]*STATES[(ORd_num_of_states * offset) + d]*( ALGEBRAIC[(ORd_num_of_algebraic * offset) + f]*(1.00000 - STATES[(ORd_num_of_states * offset) + nca])+ STATES[(ORd_num_of_states * offset) + jca]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + fca]*STATES[(ORd_num_of_states * offset) + nca])+ ALGEBRAIC[(ORd_num_of_algebraic * offset) + fICaLp]*CONSTANTS[(ORd_num_of_constants * offset) + PCap]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + PhiCaL]*STATES[(ORd_num_of_states * offset) + d]*( ALGEBRAIC[(ORd_num_of_algebraic * offset) + fp]*(1.00000 - STATES[(ORd_num_of_states * offset) + nca])+ STATES[(ORd_num_of_states * offset) + jca]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + fcap]*STATES[(ORd_num_of_states * offset) + nca]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + Jrel_inf_temp] = ( CONSTANTS[(ORd_num_of_constants * offset) + a_rel]*- ALGEBRAIC[(ORd_num_of_algebraic * offset) + ICaL])/(1.00000+ 1.00000*pow(1.50000/STATES[(ORd_num_of_states * offset) + cajsr], 8.00000));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + Jrel_inf] = (CONSTANTS[(ORd_num_of_constants * offset) + celltype]==2.00000 ?  ALGEBRAIC[(ORd_num_of_algebraic * offset) + Jrel_inf_temp]*1.70000 : ALGEBRAIC[(ORd_num_of_algebraic * offset) + Jrel_inf_temp]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + tau_rel_temp] = CONSTANTS[(ORd_num_of_constants * offset) + bt]/(1.00000+0.0123000/STATES[(ORd_num_of_states * offset) + cajsr]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + tau_rel] = (ALGEBRAIC[(ORd_num_of_algebraic * offset) + tau_rel_temp]<0.00100000 ? 0.00100000 : ALGEBRAIC[(ORd_num_of_algebraic * offset) + tau_rel_temp]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + Jrel_temp] = ( CONSTANTS[(ORd_num_of_constants * offset) + a_relp]*- ALGEBRAIC[(ORd_num_of_algebraic * offset) + ICaL])/(1.00000+pow(1.50000/STATES[(ORd_num_of_states * offset) + cajsr], 8.00000));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + Jrel_infp] = (CONSTANTS[(ORd_num_of_constants * offset) + celltype]==2.00000 ?  ALGEBRAIC[(ORd_num_of_algebraic * offset) + Jrel_temp]*1.70000 : ALGEBRAIC[(ORd_num_of_algebraic * offset) + Jrel_temp]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + tau_relp_temp] = CONSTANTS[(ORd_num_of_constants * offset) + btp]/(1.00000+0.0123000/STATES[(ORd_num_of_states * offset) + cajsr]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + tau_relp] = (ALGEBRAIC[(ORd_num_of_algebraic * offset) + tau_relp_temp]<0.00100000 ? 0.00100000 : ALGEBRAIC[(ORd_num_of_algebraic * offset) + tau_relp_temp]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + EK] =  (( CONSTANTS[(ORd_num_of_constants * offset) + R]*CONSTANTS[(ORd_num_of_constants * offset) + T])/CONSTANTS[(ORd_num_of_constants * offset) + F])*log(CONSTANTS[(ORd_num_of_constants * offset) + ko]/STATES[(ORd_num_of_states * offset) + ki]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + AiF] = 1.00000/(1.00000+exp((STATES[(ORd_num_of_states * offset) + V] - 213.600)/151.200));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + AiS] = 1.00000 - ALGEBRAIC[(ORd_num_of_algebraic * offset) + AiF];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + i] =  ALGEBRAIC[(ORd_num_of_algebraic * offset) + AiF]*STATES[(ORd_num_of_states * offset) + iF]+ ALGEBRAIC[(ORd_num_of_algebraic * offset) + AiS]*STATES[(ORd_num_of_states * offset) + iS];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + ip] =  ALGEBRAIC[(ORd_num_of_algebraic * offset) + AiF]*STATES[(ORd_num_of_states * offset) + iFp]+ ALGEBRAIC[(ORd_num_of_algebraic * offset) + AiS]*STATES[(ORd_num_of_states * offset) + iSp];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + fItop] = 1.00000/(1.00000+CONSTANTS[(ORd_num_of_constants * offset) + KmCaMK]/ALGEBRAIC[(ORd_num_of_algebraic * offset) + CaMKa]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + Ito] =  CONSTANTS[(ORd_num_of_constants * offset) + Gto]*(STATES[(ORd_num_of_states * offset) + V] - ALGEBRAIC[(ORd_num_of_algebraic * offset) + EK])*( (1.00000 - ALGEBRAIC[(ORd_num_of_algebraic * offset) + fItop])*STATES[(ORd_num_of_states * offset) + a]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + i]+ ALGEBRAIC[(ORd_num_of_algebraic * offset) + fItop]*STATES[(ORd_num_of_states * offset) + ap]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + ip]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + IKr] =  CONSTANTS[(ORd_num_of_constants * offset) + GKr]* pow((CONSTANTS[(ORd_num_of_constants * offset) + ko]/5.40000), 1.0 / 2)*STATES[(ORd_num_of_states * offset) + O]*(STATES[(ORd_num_of_states * offset) + V] - ALGEBRAIC[(ORd_num_of_algebraic * offset) + EK]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + EKs] =  (( CONSTANTS[(ORd_num_of_constants * offset) + R]*CONSTANTS[(ORd_num_of_constants * offset) + T])/CONSTANTS[(ORd_num_of_constants * offset) + F])*log((CONSTANTS[(ORd_num_of_constants * offset) + ko]+ CONSTANTS[(ORd_num_of_constants * offset) + PKNa]*CONSTANTS[(ORd_num_of_constants * offset) + nao])/(STATES[(ORd_num_of_states * offset) + ki]+ CONSTANTS[(ORd_num_of_constants * offset) + PKNa]*STATES[(ORd_num_of_states * offset) + nai]));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + KsCa] = 1.00000+0.600000/(1.00000+pow(3.80000e-05/STATES[(ORd_num_of_states * offset) + cai], 1.40000));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + IKs] =  CONSTANTS[(ORd_num_of_constants * offset) + GKs]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + KsCa]*STATES[(ORd_num_of_states * offset) + xs1]*STATES[(ORd_num_of_states * offset) + xs2]*(STATES[(ORd_num_of_states * offset) + V] - ALGEBRAIC[(ORd_num_of_algebraic * offset) + EKs]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + rk1] = 1.00000/(1.00000+exp(((STATES[(ORd_num_of_states * offset) + V]+105.800) -  2.60000*CONSTANTS[(ORd_num_of_constants * offset) + ko])/9.49300));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + IK1] =  CONSTANTS[(ORd_num_of_constants * offset) + GK1]* pow(CONSTANTS[(ORd_num_of_constants * offset) + ko], 1.0 / 2)*ALGEBRAIC[(ORd_num_of_algebraic * offset) + rk1]*STATES[(ORd_num_of_states * offset) + xk1]*(STATES[(ORd_num_of_states * offset) + V] - ALGEBRAIC[(ORd_num_of_algebraic * offset) + EK]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + Knao] =  CONSTANTS[(ORd_num_of_constants * offset) + Knao0]*exp(( (1.00000 - CONSTANTS[(ORd_num_of_constants * offset) + delta])*STATES[(ORd_num_of_states * offset) + V]*CONSTANTS[(ORd_num_of_constants * offset) + F])/( 3.00000*CONSTANTS[(ORd_num_of_constants * offset) + R]*CONSTANTS[(ORd_num_of_constants * offset) + T]));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + a3] = ( CONSTANTS[(ORd_num_of_constants * offset) + k3p]*pow(CONSTANTS[(ORd_num_of_constants * offset) + ko]/CONSTANTS[(ORd_num_of_constants * offset) + Kko], 2.00000))/((pow(1.00000+CONSTANTS[(ORd_num_of_constants * offset) + nao]/ALGEBRAIC[(ORd_num_of_algebraic * offset) + Knao], 3.00000)+pow(1.00000+CONSTANTS[(ORd_num_of_constants * offset) + ko]/CONSTANTS[(ORd_num_of_constants * offset) + Kko], 2.00000)) - 1.00000);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + P] = CONSTANTS[(ORd_num_of_constants * offset) + eP]/(1.00000+CONSTANTS[(ORd_num_of_constants * offset) + H]/CONSTANTS[(ORd_num_of_constants * offset) + Khp]+STATES[(ORd_num_of_states * offset) + nai]/CONSTANTS[(ORd_num_of_constants * offset) + Knap]+STATES[(ORd_num_of_states * offset) + ki]/CONSTANTS[(ORd_num_of_constants * offset) + Kxkur]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + b3] = ( CONSTANTS[(ORd_num_of_constants * offset) + k3m]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + P]*CONSTANTS[(ORd_num_of_constants * offset) + H])/(1.00000+CONSTANTS[(ORd_num_of_constants * offset) + MgATP]/CONSTANTS[(ORd_num_of_constants * offset) + Kmgatp]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + Knai] =  CONSTANTS[(ORd_num_of_constants * offset) + Knai0]*exp(( CONSTANTS[(ORd_num_of_constants * offset) + delta]*STATES[(ORd_num_of_states * offset) + V]*CONSTANTS[(ORd_num_of_constants * offset) + F])/( 3.00000*CONSTANTS[(ORd_num_of_constants * offset) + R]*CONSTANTS[(ORd_num_of_constants * offset) + T]));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + a1] = ( CONSTANTS[(ORd_num_of_constants * offset) + k1p]*pow(STATES[(ORd_num_of_states * offset) + nai]/ALGEBRAIC[(ORd_num_of_algebraic * offset) + Knai], 3.00000))/((pow(1.00000+STATES[(ORd_num_of_states * offset) + nai]/ALGEBRAIC[(ORd_num_of_algebraic * offset) + Knai], 3.00000)+pow(1.00000+STATES[(ORd_num_of_states * offset) + ki]/CONSTANTS[(ORd_num_of_constants * offset) + Kki], 2.00000)) - 1.00000);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + b2] = ( CONSTANTS[(ORd_num_of_constants * offset) + k2m]*pow(CONSTANTS[(ORd_num_of_constants * offset) + nao]/ALGEBRAIC[(ORd_num_of_algebraic * offset) + Knao], 3.00000))/((pow(1.00000+CONSTANTS[(ORd_num_of_constants * offset) + nao]/ALGEBRAIC[(ORd_num_of_algebraic * offset) + Knao], 3.00000)+pow(1.00000+CONSTANTS[(ORd_num_of_constants * offset) + ko]/CONSTANTS[(ORd_num_of_constants * offset) + Kko], 2.00000)) - 1.00000);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + b4] = ( CONSTANTS[(ORd_num_of_constants * offset) + k4m]*pow(STATES[(ORd_num_of_states * offset) + ki]/CONSTANTS[(ORd_num_of_constants * offset) + Kki], 2.00000))/((pow(1.00000+STATES[(ORd_num_of_states * offset) + nai]/ALGEBRAIC[(ORd_num_of_algebraic * offset) + Knai], 3.00000)+pow(1.00000+STATES[(ORd_num_of_states * offset) + ki]/CONSTANTS[(ORd_num_of_constants * offset) + Kki], 2.00000)) - 1.00000);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + x1] =  CONSTANTS[(ORd_num_of_constants * offset) + a4]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + a1]*CONSTANTS[(ORd_num_of_constants * offset) + a2]+ ALGEBRAIC[(ORd_num_of_algebraic * offset) + b2]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + b4]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + b3]+ CONSTANTS[(ORd_num_of_constants * offset) + a2]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + b4]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + b3]+ ALGEBRAIC[(ORd_num_of_algebraic * offset) + b3]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + a1]*CONSTANTS[(ORd_num_of_constants * offset) + a2];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + x2] =  ALGEBRAIC[(ORd_num_of_algebraic * offset) + b2]*CONSTANTS[(ORd_num_of_constants * offset) + b1]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + b4]+ ALGEBRAIC[(ORd_num_of_algebraic * offset) + a1]*CONSTANTS[(ORd_num_of_constants * offset) + a2]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + a3]+ ALGEBRAIC[(ORd_num_of_algebraic * offset) + a3]*CONSTANTS[(ORd_num_of_constants * offset) + b1]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + b4]+ CONSTANTS[(ORd_num_of_constants * offset) + a2]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + a3]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + b4];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + x3] =  CONSTANTS[(ORd_num_of_constants * offset) + a2]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + a3]*CONSTANTS[(ORd_num_of_constants * offset) + a4]+ ALGEBRAIC[(ORd_num_of_algebraic * offset) + b3]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + b2]*CONSTANTS[(ORd_num_of_constants * offset) + b1]+ ALGEBRAIC[(ORd_num_of_algebraic * offset) + b2]*CONSTANTS[(ORd_num_of_constants * offset) + b1]*CONSTANTS[(ORd_num_of_constants * offset) + a4]+ ALGEBRAIC[(ORd_num_of_algebraic * offset) + a3]*CONSTANTS[(ORd_num_of_constants * offset) + a4]*CONSTANTS[(ORd_num_of_constants * offset) + b1];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + x4] =  ALGEBRAIC[(ORd_num_of_algebraic * offset) + b4]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + b3]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + b2]+ ALGEBRAIC[(ORd_num_of_algebraic * offset) + a3]*CONSTANTS[(ORd_num_of_constants * offset) + a4]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + a1]+ ALGEBRAIC[(ORd_num_of_algebraic * offset) + b2]*CONSTANTS[(ORd_num_of_constants * offset) + a4]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + a1]+ ALGEBRAIC[(ORd_num_of_algebraic * offset) + b3]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + b2]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + a1];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + E1] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + x1]/(ALGEBRAIC[(ORd_num_of_algebraic * offset) + x1]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + x2]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + x3]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + x4]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + E2] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + x2]/(ALGEBRAIC[(ORd_num_of_algebraic * offset) + x1]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + x2]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + x3]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + x4]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + JnakNa] =  3.00000*( ALGEBRAIC[(ORd_num_of_algebraic * offset) + E1]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + a3] -  ALGEBRAIC[(ORd_num_of_algebraic * offset) + E2]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + b3]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + E3] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + x3]/(ALGEBRAIC[(ORd_num_of_algebraic * offset) + x1]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + x2]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + x3]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + x4]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + E4] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + x4]/(ALGEBRAIC[(ORd_num_of_algebraic * offset) + x1]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + x2]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + x3]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + x4]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + JnakK] =  2.00000*( ALGEBRAIC[(ORd_num_of_algebraic * offset) + E4]*CONSTANTS[(ORd_num_of_constants * offset) + b1] -  ALGEBRAIC[(ORd_num_of_algebraic * offset) + E3]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + a1]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + INaK] =  CONSTANTS[(ORd_num_of_constants * offset) + Pnak]*( CONSTANTS[(ORd_num_of_constants * offset) + zna]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + JnakNa]+ CONSTANTS[(ORd_num_of_constants * offset) + zk]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + JnakK]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + xkb] = 1.00000/(1.00000+exp(- (STATES[(ORd_num_of_states * offset) + V] - 14.4800)/18.3400));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + IKb] =  CONSTANTS[(ORd_num_of_constants * offset) + GKb]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + xkb]*(STATES[(ORd_num_of_states * offset) + V] - ALGEBRAIC[(ORd_num_of_algebraic * offset) + EK]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + Istim] = (TIME>=CONSTANTS[(ORd_num_of_constants * offset) + stim_start]&&TIME<=CONSTANTS[(ORd_num_of_constants * offset) + stim_end]&&(TIME - CONSTANTS[(ORd_num_of_constants * offset) + stim_start]) -  floor((TIME - CONSTANTS[(ORd_num_of_constants * offset) + stim_start])/CONSTANTS[(ORd_num_of_constants * offset) + BCL])*CONSTANTS[(ORd_num_of_constants * offset) + BCL]<=CONSTANTS[(ORd_num_of_constants * offset) + duration] ? CONSTANTS[(ORd_num_of_constants * offset) + amp] : 0.000000);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + JdiffK] = (STATES[(ORd_num_of_states * offset) + kss] - STATES[(ORd_num_of_states * offset) + ki])/2.00000;
ALGEBRAIC[(ORd_num_of_algebraic * offset) + A_3] = ( 0.750000*CONSTANTS[(ORd_num_of_constants * offset) + ffrt]*( STATES[(ORd_num_of_states * offset) + kss]*exp(ALGEBRAIC[(ORd_num_of_algebraic * offset) + vfrt]) - CONSTANTS[(ORd_num_of_constants * offset) + ko]))/CONSTANTS[(ORd_num_of_constants * offset) + B_3];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + U_3] =  CONSTANTS[(ORd_num_of_constants * offset) + B_3]*(STATES[(ORd_num_of_states * offset) + V] - CONSTANTS[(ORd_num_of_constants * offset) + v0_CaL]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + PhiCaK] = (- 1.00000e-07<=ALGEBRAIC[(ORd_num_of_algebraic * offset) + U_3]&&ALGEBRAIC[(ORd_num_of_algebraic * offset) + U_3]<=1.00000e-07 ?  ALGEBRAIC[(ORd_num_of_algebraic * offset) + A_3]*(1.00000 -  0.500000*ALGEBRAIC[(ORd_num_of_algebraic * offset) + U_3]) : ( ALGEBRAIC[(ORd_num_of_algebraic * offset) + A_3]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + U_3])/(exp(ALGEBRAIC[(ORd_num_of_algebraic * offset) + U_3]) - 1.00000));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + ICaK] =  (1.00000 - ALGEBRAIC[(ORd_num_of_algebraic * offset) + fICaLp])*CONSTANTS[(ORd_num_of_constants * offset) + PCaK]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + PhiCaK]*STATES[(ORd_num_of_states * offset) + d]*( ALGEBRAIC[(ORd_num_of_algebraic * offset) + f]*(1.00000 - STATES[(ORd_num_of_states * offset) + nca])+ STATES[(ORd_num_of_states * offset) + jca]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + fca]*STATES[(ORd_num_of_states * offset) + nca])+ ALGEBRAIC[(ORd_num_of_algebraic * offset) + fICaLp]*CONSTANTS[(ORd_num_of_constants * offset) + PCaKp]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + PhiCaK]*STATES[(ORd_num_of_states * offset) + d]*( ALGEBRAIC[(ORd_num_of_algebraic * offset) + fp]*(1.00000 - STATES[(ORd_num_of_states * offset) + nca])+ STATES[(ORd_num_of_states * offset) + jca]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + fcap]*STATES[(ORd_num_of_states * offset) + nca]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + ENa] =  (( CONSTANTS[(ORd_num_of_constants * offset) + R]*CONSTANTS[(ORd_num_of_constants * offset) + T])/CONSTANTS[(ORd_num_of_constants * offset) + F])*log(CONSTANTS[(ORd_num_of_constants * offset) + nao]/STATES[(ORd_num_of_states * offset) + nai]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + h] =  CONSTANTS[(ORd_num_of_constants * offset) + Ahf]*STATES[(ORd_num_of_states * offset) + hf]+ CONSTANTS[(ORd_num_of_constants * offset) + Ahs]*STATES[(ORd_num_of_states * offset) + hs];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + hp] =  CONSTANTS[(ORd_num_of_constants * offset) + Ahf]*STATES[(ORd_num_of_states * offset) + hf]+ CONSTANTS[(ORd_num_of_constants * offset) + Ahs]*STATES[(ORd_num_of_states * offset) + hsp];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + fINap] = 1.00000/(1.00000+CONSTANTS[(ORd_num_of_constants * offset) + KmCaMK]/ALGEBRAIC[(ORd_num_of_algebraic * offset) + CaMKa]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + INa] =  CONSTANTS[(ORd_num_of_constants * offset) + GNa]*(STATES[(ORd_num_of_states * offset) + V] - ALGEBRAIC[(ORd_num_of_algebraic * offset) + ENa])*pow(STATES[(ORd_num_of_states * offset) + m], 3.00000)*( (1.00000 - ALGEBRAIC[(ORd_num_of_algebraic * offset) + fINap])*ALGEBRAIC[(ORd_num_of_algebraic * offset) + h]*STATES[(ORd_num_of_states * offset) + j]+ ALGEBRAIC[(ORd_num_of_algebraic * offset) + fINap]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + hp]*STATES[(ORd_num_of_states * offset) + jp]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + fINaLp] = 1.00000/(1.00000+CONSTANTS[(ORd_num_of_constants * offset) + KmCaMK]/ALGEBRAIC[(ORd_num_of_algebraic * offset) + CaMKa]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + INaL] =  CONSTANTS[(ORd_num_of_constants * offset) + GNaL]*(STATES[(ORd_num_of_states * offset) + V] - ALGEBRAIC[(ORd_num_of_algebraic * offset) + ENa])*STATES[(ORd_num_of_states * offset) + mL]*( (1.00000 - ALGEBRAIC[(ORd_num_of_algebraic * offset) + fINaLp])*STATES[(ORd_num_of_states * offset) + hL]+ ALGEBRAIC[(ORd_num_of_algebraic * offset) + fINaLp]*STATES[(ORd_num_of_states * offset) + hLp]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + allo_i] = 1.00000/(1.00000+pow(CONSTANTS[(ORd_num_of_constants * offset) + KmCaAct]/STATES[(ORd_num_of_states * offset) + cai], 2.00000));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + hna] = exp(( CONSTANTS[(ORd_num_of_constants * offset) + qna]*STATES[(ORd_num_of_states * offset) + V]*CONSTANTS[(ORd_num_of_constants * offset) + F])/( CONSTANTS[(ORd_num_of_constants * offset) + R]*CONSTANTS[(ORd_num_of_constants * offset) + T]));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + h7_i] = 1.00000+ (CONSTANTS[(ORd_num_of_constants * offset) + nao]/CONSTANTS[(ORd_num_of_constants * offset) + kna3])*(1.00000+1.00000/ALGEBRAIC[(ORd_num_of_algebraic * offset) + hna]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + h8_i] = CONSTANTS[(ORd_num_of_constants * offset) + nao]/( CONSTANTS[(ORd_num_of_constants * offset) + kna3]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + hna]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + h7_i]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + k3pp_i] =  ALGEBRAIC[(ORd_num_of_algebraic * offset) + h8_i]*CONSTANTS[(ORd_num_of_constants * offset) + wnaca];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + h1_i] = 1.00000+ (STATES[(ORd_num_of_states * offset) + nai]/CONSTANTS[(ORd_num_of_constants * offset) + kna3])*(1.00000+ALGEBRAIC[(ORd_num_of_algebraic * offset) + hna]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + h2_i] = ( STATES[(ORd_num_of_states * offset) + nai]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + hna])/( CONSTANTS[(ORd_num_of_constants * offset) + kna3]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + h1_i]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + k4pp_i] =  ALGEBRAIC[(ORd_num_of_algebraic * offset) + h2_i]*CONSTANTS[(ORd_num_of_constants * offset) + wnaca];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + h4_i] = 1.00000+ (STATES[(ORd_num_of_states * offset) + nai]/CONSTANTS[(ORd_num_of_constants * offset) + kna1])*(1.00000+STATES[(ORd_num_of_states * offset) + nai]/CONSTANTS[(ORd_num_of_constants * offset) + kna2]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + h5_i] = ( STATES[(ORd_num_of_states * offset) + nai]*STATES[(ORd_num_of_states * offset) + nai])/( ALGEBRAIC[(ORd_num_of_algebraic * offset) + h4_i]*CONSTANTS[(ORd_num_of_constants * offset) + kna1]*CONSTANTS[(ORd_num_of_constants * offset) + kna2]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + k7_i] =  ALGEBRAIC[(ORd_num_of_algebraic * offset) + h5_i]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + h2_i]*CONSTANTS[(ORd_num_of_constants * offset) + wna];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + k8_i] =  ALGEBRAIC[(ORd_num_of_algebraic * offset) + h8_i]*CONSTANTS[(ORd_num_of_constants * offset) + h11_i]*CONSTANTS[(ORd_num_of_constants * offset) + wna];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + h9_i] = 1.00000/ALGEBRAIC[(ORd_num_of_algebraic * offset) + h7_i];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + k3p_i] =  ALGEBRAIC[(ORd_num_of_algebraic * offset) + h9_i]*CONSTANTS[(ORd_num_of_constants * offset) + wca];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + k3_i] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + k3p_i]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + k3pp_i];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + hca] = exp(( CONSTANTS[(ORd_num_of_constants * offset) + qca]*STATES[(ORd_num_of_states * offset) + V]*CONSTANTS[(ORd_num_of_constants * offset) + F])/( CONSTANTS[(ORd_num_of_constants * offset) + R]*CONSTANTS[(ORd_num_of_constants * offset) + T]));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + h3_i] = 1.00000/ALGEBRAIC[(ORd_num_of_algebraic * offset) + h1_i];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + k4p_i] = ( ALGEBRAIC[(ORd_num_of_algebraic * offset) + h3_i]*CONSTANTS[(ORd_num_of_constants * offset) + wca])/ALGEBRAIC[(ORd_num_of_algebraic * offset) + hca];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + k4_i] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + k4p_i]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + k4pp_i];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + h6_i] = 1.00000/ALGEBRAIC[(ORd_num_of_algebraic * offset) + h4_i];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + k6_i] =  ALGEBRAIC[(ORd_num_of_algebraic * offset) + h6_i]*STATES[(ORd_num_of_states * offset) + cai]*CONSTANTS[(ORd_num_of_constants * offset) + kcaon];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + x1_i] =  CONSTANTS[(ORd_num_of_constants * offset) + k2_i]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + k4_i]*(ALGEBRAIC[(ORd_num_of_algebraic * offset) + k7_i]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + k6_i])+ CONSTANTS[(ORd_num_of_constants * offset) + k5_i]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + k7_i]*(CONSTANTS[(ORd_num_of_constants * offset) + k2_i]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + k3_i]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + x2_i] =  CONSTANTS[(ORd_num_of_constants * offset) + k1_i]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + k7_i]*(ALGEBRAIC[(ORd_num_of_algebraic * offset) + k4_i]+CONSTANTS[(ORd_num_of_constants * offset) + k5_i])+ ALGEBRAIC[(ORd_num_of_algebraic * offset) + k4_i]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + k6_i]*(CONSTANTS[(ORd_num_of_constants * offset) + k1_i]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + k8_i]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + x3_i] =  CONSTANTS[(ORd_num_of_constants * offset) + k1_i]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + k3_i]*(ALGEBRAIC[(ORd_num_of_algebraic * offset) + k7_i]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + k6_i])+ ALGEBRAIC[(ORd_num_of_algebraic * offset) + k8_i]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + k6_i]*(CONSTANTS[(ORd_num_of_constants * offset) + k2_i]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + k3_i]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + x4_i] =  CONSTANTS[(ORd_num_of_constants * offset) + k2_i]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + k8_i]*(ALGEBRAIC[(ORd_num_of_algebraic * offset) + k4_i]+CONSTANTS[(ORd_num_of_constants * offset) + k5_i])+ ALGEBRAIC[(ORd_num_of_algebraic * offset) + k3_i]*CONSTANTS[(ORd_num_of_constants * offset) + k5_i]*(CONSTANTS[(ORd_num_of_constants * offset) + k1_i]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + k8_i]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + E1_i] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + x1_i]/(ALGEBRAIC[(ORd_num_of_algebraic * offset) + x1_i]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + x2_i]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + x3_i]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + x4_i]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + E2_i] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + x2_i]/(ALGEBRAIC[(ORd_num_of_algebraic * offset) + x1_i]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + x2_i]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + x3_i]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + x4_i]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + E3_i] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + x3_i]/(ALGEBRAIC[(ORd_num_of_algebraic * offset) + x1_i]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + x2_i]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + x3_i]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + x4_i]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + E4_i] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + x4_i]/(ALGEBRAIC[(ORd_num_of_algebraic * offset) + x1_i]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + x2_i]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + x3_i]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + x4_i]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + JncxNa_i] = ( 3.00000*( ALGEBRAIC[(ORd_num_of_algebraic * offset) + E4_i]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + k7_i] -  ALGEBRAIC[(ORd_num_of_algebraic * offset) + E1_i]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + k8_i])+ ALGEBRAIC[(ORd_num_of_algebraic * offset) + E3_i]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + k4pp_i]) -  ALGEBRAIC[(ORd_num_of_algebraic * offset) + E2_i]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + k3pp_i];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + JncxCa_i] =  ALGEBRAIC[(ORd_num_of_algebraic * offset) + E2_i]*CONSTANTS[(ORd_num_of_constants * offset) + k2_i] -  ALGEBRAIC[(ORd_num_of_algebraic * offset) + E1_i]*CONSTANTS[(ORd_num_of_constants * offset) + k1_i];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + INaCa_i] =  0.800000*CONSTANTS[(ORd_num_of_constants * offset) + Gncx]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + allo_i]*( CONSTANTS[(ORd_num_of_constants * offset) + zna]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + JncxNa_i]+ CONSTANTS[(ORd_num_of_constants * offset) + zca]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + JncxCa_i]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + A_Nab] = ( CONSTANTS[(ORd_num_of_constants * offset) + PNab]*CONSTANTS[(ORd_num_of_constants * offset) + ffrt]*( STATES[(ORd_num_of_states * offset) + nai]*exp(ALGEBRAIC[(ORd_num_of_algebraic * offset) + vfrt]) - CONSTANTS[(ORd_num_of_constants * offset) + nao]))/CONSTANTS[(ORd_num_of_constants * offset) + B_Nab];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + U_Nab] =  CONSTANTS[(ORd_num_of_constants * offset) + B_Nab]*(STATES[(ORd_num_of_states * offset) + V] - CONSTANTS[(ORd_num_of_constants * offset) + v0_Nab]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + INab] = (- 1.00000e-07<=ALGEBRAIC[(ORd_num_of_algebraic * offset) + U_Nab]&&ALGEBRAIC[(ORd_num_of_algebraic * offset) + U_Nab]<=1.00000e-07 ?  ALGEBRAIC[(ORd_num_of_algebraic * offset) + A_Nab]*(1.00000 -  0.500000*ALGEBRAIC[(ORd_num_of_algebraic * offset) + U_Nab]) : ( ALGEBRAIC[(ORd_num_of_algebraic * offset) + A_Nab]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + U_Nab])/(exp(ALGEBRAIC[(ORd_num_of_algebraic * offset) + U_Nab]) - 1.00000));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + JdiffNa] = (STATES[(ORd_num_of_states * offset) + nass] - STATES[(ORd_num_of_states * offset) + nai])/2.00000;
ALGEBRAIC[(ORd_num_of_algebraic * offset) + A_2] = ( 0.750000*CONSTANTS[(ORd_num_of_constants * offset) + ffrt]*( STATES[(ORd_num_of_states * offset) + nass]*exp(ALGEBRAIC[(ORd_num_of_algebraic * offset) + vfrt]) - CONSTANTS[(ORd_num_of_constants * offset) + nao]))/CONSTANTS[(ORd_num_of_constants * offset) + B_2];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + U_2] =  CONSTANTS[(ORd_num_of_constants * offset) + B_2]*(STATES[(ORd_num_of_states * offset) + V] - CONSTANTS[(ORd_num_of_constants * offset) + v0_CaL]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + PhiCaNa] = (- 1.00000e-07<=ALGEBRAIC[(ORd_num_of_algebraic * offset) + U_2]&&ALGEBRAIC[(ORd_num_of_algebraic * offset) + U_2]<=1.00000e-07 ?  ALGEBRAIC[(ORd_num_of_algebraic * offset) + A_2]*(1.00000 -  0.500000*ALGEBRAIC[(ORd_num_of_algebraic * offset) + U_2]) : ( ALGEBRAIC[(ORd_num_of_algebraic * offset) + A_2]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + U_2])/(exp(ALGEBRAIC[(ORd_num_of_algebraic * offset) + U_2]) - 1.00000));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + ICaNa] =  (1.00000 - ALGEBRAIC[(ORd_num_of_algebraic * offset) + fICaLp])*CONSTANTS[(ORd_num_of_constants * offset) + PCaNa]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + PhiCaNa]*STATES[(ORd_num_of_states * offset) + d]*( ALGEBRAIC[(ORd_num_of_algebraic * offset) + f]*(1.00000 - STATES[(ORd_num_of_states * offset) + nca])+ STATES[(ORd_num_of_states * offset) + jca]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + fca]*STATES[(ORd_num_of_states * offset) + nca])+ ALGEBRAIC[(ORd_num_of_algebraic * offset) + fICaLp]*CONSTANTS[(ORd_num_of_constants * offset) + PCaNap]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + PhiCaNa]*STATES[(ORd_num_of_states * offset) + d]*( ALGEBRAIC[(ORd_num_of_algebraic * offset) + fp]*(1.00000 - STATES[(ORd_num_of_states * offset) + nca])+ STATES[(ORd_num_of_states * offset) + jca]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + fcap]*STATES[(ORd_num_of_states * offset) + nca]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + allo_ss] = 1.00000/(1.00000+pow(CONSTANTS[(ORd_num_of_constants * offset) + KmCaAct]/STATES[(ORd_num_of_states * offset) + cass], 2.00000));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + h7_ss] = 1.00000+ (CONSTANTS[(ORd_num_of_constants * offset) + nao]/CONSTANTS[(ORd_num_of_constants * offset) + kna3])*(1.00000+1.00000/ALGEBRAIC[(ORd_num_of_algebraic * offset) + hna]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + h8_ss] = CONSTANTS[(ORd_num_of_constants * offset) + nao]/( CONSTANTS[(ORd_num_of_constants * offset) + kna3]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + hna]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + h7_ss]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + k3pp_ss] =  ALGEBRAIC[(ORd_num_of_algebraic * offset) + h8_ss]*CONSTANTS[(ORd_num_of_constants * offset) + wnaca];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + h1_ss] = 1.00000+ (STATES[(ORd_num_of_states * offset) + nass]/CONSTANTS[(ORd_num_of_constants * offset) + kna3])*(1.00000+ALGEBRAIC[(ORd_num_of_algebraic * offset) + hna]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + h2_ss] = ( STATES[(ORd_num_of_states * offset) + nass]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + hna])/( CONSTANTS[(ORd_num_of_constants * offset) + kna3]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + h1_ss]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + k4pp_ss] =  ALGEBRAIC[(ORd_num_of_algebraic * offset) + h2_ss]*CONSTANTS[(ORd_num_of_constants * offset) + wnaca];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + h4_ss] = 1.00000+ (STATES[(ORd_num_of_states * offset) + nass]/CONSTANTS[(ORd_num_of_constants * offset) + kna1])*(1.00000+STATES[(ORd_num_of_states * offset) + nass]/CONSTANTS[(ORd_num_of_constants * offset) + kna2]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + h5_ss] = ( STATES[(ORd_num_of_states * offset) + nass]*STATES[(ORd_num_of_states * offset) + nass])/( ALGEBRAIC[(ORd_num_of_algebraic * offset) + h4_ss]*CONSTANTS[(ORd_num_of_constants * offset) + kna1]*CONSTANTS[(ORd_num_of_constants * offset) + kna2]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + k7_ss] =  ALGEBRAIC[(ORd_num_of_algebraic * offset) + h5_ss]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + h2_ss]*CONSTANTS[(ORd_num_of_constants * offset) + wna];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + k8_ss] =  ALGEBRAIC[(ORd_num_of_algebraic * offset) + h8_ss]*CONSTANTS[(ORd_num_of_constants * offset) + h11_ss]*CONSTANTS[(ORd_num_of_constants * offset) + wna];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + h9_ss] = 1.00000/ALGEBRAIC[(ORd_num_of_algebraic * offset) + h7_ss];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + k3p_ss] =  ALGEBRAIC[(ORd_num_of_algebraic * offset) + h9_ss]*CONSTANTS[(ORd_num_of_constants * offset) + wca];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + k3_ss] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + k3p_ss]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + k3pp_ss];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + h3_ss] = 1.00000/ALGEBRAIC[(ORd_num_of_algebraic * offset) + h1_ss];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + k4p_ss] = ( ALGEBRAIC[(ORd_num_of_algebraic * offset) + h3_ss]*CONSTANTS[(ORd_num_of_constants * offset) + wca])/ALGEBRAIC[(ORd_num_of_algebraic * offset) + hca];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + k4_ss] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + k4p_ss]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + k4pp_ss];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + h6_ss] = 1.00000/ALGEBRAIC[(ORd_num_of_algebraic * offset) + h4_ss];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + k6_ss] =  ALGEBRAIC[(ORd_num_of_algebraic * offset) + h6_ss]*STATES[(ORd_num_of_states * offset) + cass]*CONSTANTS[(ORd_num_of_constants * offset) + kcaon];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + x1_ss] =  CONSTANTS[(ORd_num_of_constants * offset) + k2_ss]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + k4_ss]*(ALGEBRAIC[(ORd_num_of_algebraic * offset) + k7_ss]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + k6_ss])+ CONSTANTS[(ORd_num_of_constants * offset) + k5_ss]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + k7_ss]*(CONSTANTS[(ORd_num_of_constants * offset) + k2_ss]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + k3_ss]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + x2_ss] =  CONSTANTS[(ORd_num_of_constants * offset) + k1_ss]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + k7_ss]*(ALGEBRAIC[(ORd_num_of_algebraic * offset) + k4_ss]+CONSTANTS[(ORd_num_of_constants * offset) + k5_ss])+ ALGEBRAIC[(ORd_num_of_algebraic * offset) + k4_ss]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + k6_ss]*(CONSTANTS[(ORd_num_of_constants * offset) + k1_ss]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + k8_ss]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + x3_ss] =  CONSTANTS[(ORd_num_of_constants * offset) + k1_ss]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + k3_ss]*(ALGEBRAIC[(ORd_num_of_algebraic * offset) + k7_ss]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + k6_ss])+ ALGEBRAIC[(ORd_num_of_algebraic * offset) + k8_ss]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + k6_ss]*(CONSTANTS[(ORd_num_of_constants * offset) + k2_ss]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + k3_ss]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + x4_ss] =  CONSTANTS[(ORd_num_of_constants * offset) + k2_ss]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + k8_ss]*(ALGEBRAIC[(ORd_num_of_algebraic * offset) + k4_ss]+CONSTANTS[(ORd_num_of_constants * offset) + k5_ss])+ ALGEBRAIC[(ORd_num_of_algebraic * offset) + k3_ss]*CONSTANTS[(ORd_num_of_constants * offset) + k5_ss]*(CONSTANTS[(ORd_num_of_constants * offset) + k1_ss]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + k8_ss]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + E1_ss] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + x1_ss]/(ALGEBRAIC[(ORd_num_of_algebraic * offset) + x1_ss]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + x2_ss]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + x3_ss]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + x4_ss]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + E2_ss] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + x2_ss]/(ALGEBRAIC[(ORd_num_of_algebraic * offset) + x1_ss]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + x2_ss]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + x3_ss]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + x4_ss]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + E3_ss] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + x3_ss]/(ALGEBRAIC[(ORd_num_of_algebraic * offset) + x1_ss]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + x2_ss]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + x3_ss]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + x4_ss]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + E4_ss] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + x4_ss]/(ALGEBRAIC[(ORd_num_of_algebraic * offset) + x1_ss]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + x2_ss]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + x3_ss]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + x4_ss]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + JncxNa_ss] = ( 3.00000*( ALGEBRAIC[(ORd_num_of_algebraic * offset) + E4_ss]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + k7_ss] -  ALGEBRAIC[(ORd_num_of_algebraic * offset) + E1_ss]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + k8_ss])+ ALGEBRAIC[(ORd_num_of_algebraic * offset) + E3_ss]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + k4pp_ss]) -  ALGEBRAIC[(ORd_num_of_algebraic * offset) + E2_ss]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + k3pp_ss];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + JncxCa_ss] =  ALGEBRAIC[(ORd_num_of_algebraic * offset) + E2_ss]*CONSTANTS[(ORd_num_of_constants * offset) + k2_ss] -  ALGEBRAIC[(ORd_num_of_algebraic * offset) + E1_ss]*CONSTANTS[(ORd_num_of_constants * offset) + k1_ss];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + INaCa_ss] =  0.200000*CONSTANTS[(ORd_num_of_constants * offset) + Gncx]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + allo_ss]*( CONSTANTS[(ORd_num_of_constants * offset) + zna]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + JncxNa_ss]+ CONSTANTS[(ORd_num_of_constants * offset) + zca]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + JncxCa_ss]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + IpCa] = ( CONSTANTS[(ORd_num_of_constants * offset) + GpCa]*STATES[(ORd_num_of_states * offset) + cai])/(CONSTANTS[(ORd_num_of_constants * offset) + KmCap]+STATES[(ORd_num_of_states * offset) + cai]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + A_Cab] = ( CONSTANTS[(ORd_num_of_constants * offset) + PCab]*4.00000*CONSTANTS[(ORd_num_of_constants * offset) + ffrt]*( STATES[(ORd_num_of_states * offset) + cai]*exp( 2.00000*ALGEBRAIC[(ORd_num_of_algebraic * offset) + vfrt]) -  0.341000*CONSTANTS[(ORd_num_of_constants * offset) + cao]))/CONSTANTS[(ORd_num_of_constants * offset) + B_Cab];
ALGEBRAIC[(ORd_num_of_algebraic * offset) + U_Cab] =  CONSTANTS[(ORd_num_of_constants * offset) + B_Cab]*(STATES[(ORd_num_of_states * offset) + V] - CONSTANTS[(ORd_num_of_constants * offset) + v0_Cab]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + ICab] = (- 1.00000e-07<=ALGEBRAIC[(ORd_num_of_algebraic * offset) + U_Cab]&&ALGEBRAIC[(ORd_num_of_algebraic * offset) + U_Cab]<=1.00000e-07 ?  ALGEBRAIC[(ORd_num_of_algebraic * offset) + A_Cab]*(1.00000 -  0.500000*ALGEBRAIC[(ORd_num_of_algebraic * offset) + U_Cab]) : ( ALGEBRAIC[(ORd_num_of_algebraic * offset) + A_Cab]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + U_Cab])/(exp(ALGEBRAIC[(ORd_num_of_algebraic * offset) + U_Cab]) - 1.00000));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + Jdiff] = (STATES[(ORd_num_of_states * offset) + cass] - STATES[(ORd_num_of_states * offset) + cai])/0.200000;
ALGEBRAIC[(ORd_num_of_algebraic * offset) + fJrelp] = 1.00000/(1.00000+CONSTANTS[(ORd_num_of_constants * offset) + KmCaMK]/ALGEBRAIC[(ORd_num_of_algebraic * offset) + CaMKa]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + Jrel] =  CONSTANTS[(ORd_num_of_constants * offset) + Jrel_scaling_factor]*( (1.00000 - ALGEBRAIC[(ORd_num_of_algebraic * offset) + fJrelp])*STATES[(ORd_num_of_states * offset) + Jrelnp]+ ALGEBRAIC[(ORd_num_of_algebraic * offset) + fJrelp]*STATES[(ORd_num_of_states * offset) + Jrelp]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + Bcass] = 1.00000/(1.00000+( CONSTANTS[(ORd_num_of_constants * offset) + BSRmax]*CONSTANTS[(ORd_num_of_constants * offset) + KmBSR])/pow(CONSTANTS[(ORd_num_of_constants * offset) + KmBSR]+STATES[(ORd_num_of_states * offset) + cass], 2.00000)+( CONSTANTS[(ORd_num_of_constants * offset) + BSLmax]*CONSTANTS[(ORd_num_of_constants * offset) + KmBSL])/pow(CONSTANTS[(ORd_num_of_constants * offset) + KmBSL]+STATES[(ORd_num_of_states * offset) + cass], 2.00000));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + Jupnp] = ( CONSTANTS[(ORd_num_of_constants * offset) + upScale]*0.00437500*STATES[(ORd_num_of_states * offset) + cai])/(STATES[(ORd_num_of_states * offset) + cai]+0.000920000);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + Jupp] = ( CONSTANTS[(ORd_num_of_constants * offset) + upScale]*2.75000*0.00437500*STATES[(ORd_num_of_states * offset) + cai])/((STATES[(ORd_num_of_states * offset) + cai]+0.000920000) - 0.000170000);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + fJupp] = 1.00000/(1.00000+CONSTANTS[(ORd_num_of_constants * offset) + KmCaMK]/ALGEBRAIC[(ORd_num_of_algebraic * offset) + CaMKa]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + Jleak] = ( 0.00393750*STATES[(ORd_num_of_states * offset) + cansr])/15.0000;
ALGEBRAIC[(ORd_num_of_algebraic * offset) + Jup] =  CONSTANTS[(ORd_num_of_constants * offset) + Jup_b]*(( (1.00000 - ALGEBRAIC[(ORd_num_of_algebraic * offset) + fJupp])*ALGEBRAIC[(ORd_num_of_algebraic * offset) + Jupnp]+ ALGEBRAIC[(ORd_num_of_algebraic * offset) + fJupp]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + Jupp]) - ALGEBRAIC[(ORd_num_of_algebraic * offset) + Jleak]);
ALGEBRAIC[(ORd_num_of_algebraic * offset) + Bcai] = 1.00000/(1.00000+( CONSTANTS[(ORd_num_of_constants * offset) + cmdnmax]*CONSTANTS[(ORd_num_of_constants * offset) + kmcmdn])/pow(CONSTANTS[(ORd_num_of_constants * offset) + kmcmdn]+STATES[(ORd_num_of_states * offset) + cai], 2.00000)+( CONSTANTS[(ORd_num_of_constants * offset) + trpnmax]*CONSTANTS[(ORd_num_of_constants * offset) + kmtrpn])/pow(CONSTANTS[(ORd_num_of_constants * offset) + kmtrpn]+STATES[(ORd_num_of_states * offset) + cai], 2.00000));
ALGEBRAIC[(ORd_num_of_algebraic * offset) + Jtr] = (STATES[(ORd_num_of_states * offset) + cansr] - STATES[(ORd_num_of_states * offset) + cajsr])/100.000;
ALGEBRAIC[(ORd_num_of_algebraic * offset) + Bcajsr] = 1.00000/(1.00000+( CONSTANTS[(ORd_num_of_constants * offset) + csqnmax]*CONSTANTS[(ORd_num_of_constants * offset) + kmcsqn])/pow(CONSTANTS[(ORd_num_of_constants * offset) + kmcsqn]+STATES[(ORd_num_of_states * offset) + cajsr], 2.00000));

//RATES[D] = CONSTANTS[cnc];
RATES[(ORd_num_of_rates * offset) + D] = 0.;
RATES[(ORd_num_of_rates * offset) + IC1] = (- ( CONSTANTS[(ORd_num_of_constants * offset) +  A11]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B11]*STATES[(ORd_num_of_states * offset) + V])*STATES[(ORd_num_of_states * offset) + IC1]*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q11]))/10.0000) -  CONSTANTS[(ORd_num_of_constants * offset) +  A21]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B21]*STATES[(ORd_num_of_states * offset) + V])*STATES[(ORd_num_of_states * offset) + IC2]*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q21]))/10.0000))+ CONSTANTS[(ORd_num_of_constants * offset) +  A51]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B51]*STATES[(ORd_num_of_states * offset) + V])*STATES[(ORd_num_of_states * offset) + C1]*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q51]))/10.0000)) -  CONSTANTS[(ORd_num_of_constants * offset) +  A61]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B61]*STATES[(ORd_num_of_states * offset) + V])*STATES[(ORd_num_of_states * offset) + IC1]*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q61]))/10.0000);
RATES[(ORd_num_of_rates * offset) + IC2] = ((( CONSTANTS[(ORd_num_of_constants * offset) +  A11]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B11]*STATES[(ORd_num_of_states * offset) + V])*STATES[(ORd_num_of_states * offset) + IC1]*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q11]))/10.0000) -  CONSTANTS[(ORd_num_of_constants * offset) +  A21]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B21]*STATES[(ORd_num_of_states * offset) + V])*STATES[(ORd_num_of_states * offset) + IC2]*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q21]))/10.0000)) - ( CONSTANTS[(ORd_num_of_constants * offset) +  A3]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B3]*STATES[(ORd_num_of_states * offset) + V])*STATES[(ORd_num_of_states * offset) + IC2]*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q3]))/10.0000) -  CONSTANTS[(ORd_num_of_constants * offset) +  A4]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B4]*STATES[(ORd_num_of_states * offset) + V])*STATES[(ORd_num_of_states * offset) + IO]*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q4]))/10.0000)))+ CONSTANTS[(ORd_num_of_constants * offset) +  A52]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B52]*STATES[(ORd_num_of_states * offset) + V])*STATES[(ORd_num_of_states * offset) + C2]*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q52]))/10.0000)) -  CONSTANTS[(ORd_num_of_constants * offset) +  A62]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B62]*STATES[(ORd_num_of_states * offset) + V])*STATES[(ORd_num_of_states * offset) + IC2]*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q62]))/10.0000);
RATES[(ORd_num_of_rates * offset) + C1] = - ( CONSTANTS[(ORd_num_of_constants * offset) +  A1]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B1]*STATES[(ORd_num_of_states * offset) + V])*STATES[(ORd_num_of_states * offset) + C1]*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q1]))/10.0000) -  CONSTANTS[(ORd_num_of_constants * offset) +  A2]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B2]*STATES[(ORd_num_of_states * offset) + V])*STATES[(ORd_num_of_states * offset) + C2]*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q2]))/10.0000)) - ( CONSTANTS[(ORd_num_of_constants * offset) +  A51]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B51]*STATES[(ORd_num_of_states * offset) + V])*STATES[(ORd_num_of_states * offset) + C1]*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q51]))/10.0000) -  CONSTANTS[(ORd_num_of_constants * offset) +  A61]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B61]*STATES[(ORd_num_of_states * offset) + V])*STATES[(ORd_num_of_states * offset) + IC1]*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q61]))/10.0000));
RATES[(ORd_num_of_rates * offset) + C2] = (( CONSTANTS[(ORd_num_of_constants * offset) +  A1]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B1]*STATES[(ORd_num_of_states * offset) + V])*STATES[(ORd_num_of_states * offset) + C1]*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q1]))/10.0000) -  CONSTANTS[(ORd_num_of_constants * offset) +  A2]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B2]*STATES[(ORd_num_of_states * offset) + V])*STATES[(ORd_num_of_states * offset) + C2]*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q2]))/10.0000)) - ( CONSTANTS[(ORd_num_of_constants * offset) +  A31]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B31]*STATES[(ORd_num_of_states * offset) + V])*STATES[(ORd_num_of_states * offset) + C2]*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q31]))/10.0000) -  CONSTANTS[(ORd_num_of_constants * offset) +  A41]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B41]*STATES[(ORd_num_of_states * offset) + V])*STATES[(ORd_num_of_states * offset) + O]*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q41]))/10.0000))) - ( CONSTANTS[(ORd_num_of_constants * offset) +  A52]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B52]*STATES[(ORd_num_of_states * offset) + V])*STATES[(ORd_num_of_states * offset) + C2]*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q52]))/10.0000) -  CONSTANTS[(ORd_num_of_constants * offset) +  A62]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B62]*STATES[(ORd_num_of_states * offset) + V])*STATES[(ORd_num_of_states * offset) + IC2]*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q62]))/10.0000));
RATES[(ORd_num_of_rates * offset) + O] = (( CONSTANTS[(ORd_num_of_constants * offset) +  A31]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B31]*STATES[(ORd_num_of_states * offset) + V])*STATES[(ORd_num_of_states * offset) + C2]*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q31]))/10.0000) -  CONSTANTS[(ORd_num_of_constants * offset) +  A41]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B41]*STATES[(ORd_num_of_states * offset) + V])*STATES[(ORd_num_of_states * offset) + O]*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q41]))/10.0000)) - ( CONSTANTS[(ORd_num_of_constants * offset) +  A53]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B53]*STATES[(ORd_num_of_states * offset) + V])*STATES[(ORd_num_of_states * offset) + O]*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q53]))/10.0000) -  CONSTANTS[(ORd_num_of_constants * offset) +  A63]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B63]*STATES[(ORd_num_of_states * offset) + V])*STATES[(ORd_num_of_states * offset) + IO]*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q63]))/10.0000))) - ( (( CONSTANTS[(ORd_num_of_constants * offset) +  Kmax]*CONSTANTS[(ORd_num_of_constants * offset) +  Ku]*pow( STATES[(ORd_num_of_states * offset) + D],CONSTANTS[(ORd_num_of_constants * offset) +  n]))/(pow( STATES[(ORd_num_of_states * offset) + D],CONSTANTS[(ORd_num_of_constants * offset) +  n])+CONSTANTS[(ORd_num_of_constants * offset) +  halfmax]))*STATES[(ORd_num_of_states * offset) + O] -  CONSTANTS[(ORd_num_of_constants * offset) +  Ku]*STATES[(ORd_num_of_states * offset) + Obound]);
RATES[(ORd_num_of_rates * offset) + IO] = ((( CONSTANTS[(ORd_num_of_constants * offset) +  A3]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B3]*STATES[(ORd_num_of_states * offset) + V])*STATES[(ORd_num_of_states * offset) + IC2]*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q3]))/10.0000) -  CONSTANTS[(ORd_num_of_constants * offset) +  A4]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B4]*STATES[(ORd_num_of_states * offset) + V])*STATES[(ORd_num_of_states * offset) + IO]*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q4]))/10.0000))+ CONSTANTS[(ORd_num_of_constants * offset) +  A53]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B53]*STATES[(ORd_num_of_states * offset) + V])*STATES[(ORd_num_of_states * offset) + O]*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q53]))/10.0000)) -  CONSTANTS[(ORd_num_of_constants * offset) +  A63]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B63]*STATES[(ORd_num_of_states * offset) + V])*STATES[(ORd_num_of_states * offset) + IO]*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q63]))/10.0000)) - ( (( CONSTANTS[(ORd_num_of_constants * offset) +  Kmax]*CONSTANTS[(ORd_num_of_constants * offset) +  Ku]*pow( STATES[(ORd_num_of_states * offset) + D],CONSTANTS[(ORd_num_of_constants * offset) +  n]))/(pow( STATES[(ORd_num_of_states * offset) + D],CONSTANTS[(ORd_num_of_constants * offset) +  n])+CONSTANTS[(ORd_num_of_constants * offset) +  halfmax]))*STATES[(ORd_num_of_states * offset) + IO] -  (( CONSTANTS[(ORd_num_of_constants * offset) +  Ku]*CONSTANTS[(ORd_num_of_constants * offset) +  A53]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B53]*STATES[(ORd_num_of_states * offset) + V])*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q53]))/10.0000))/( CONSTANTS[(ORd_num_of_constants * offset) +  A63]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B63]*STATES[(ORd_num_of_states * offset) + V])*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q63]))/10.0000)))*STATES[(ORd_num_of_states * offset) + IObound]);
RATES[(ORd_num_of_rates * offset) + IObound] = (( (( CONSTANTS[(ORd_num_of_constants * offset) +  Kmax]*CONSTANTS[(ORd_num_of_constants * offset) +  Ku]*pow( STATES[(ORd_num_of_states * offset) + D],CONSTANTS[(ORd_num_of_constants * offset) +  n]))/(pow( STATES[(ORd_num_of_states * offset) + D],CONSTANTS[(ORd_num_of_constants * offset) +  n])+CONSTANTS[(ORd_num_of_constants * offset) +  halfmax]))*STATES[(ORd_num_of_states * offset) + IO] -  (( CONSTANTS[(ORd_num_of_constants * offset) +  Ku]*CONSTANTS[(ORd_num_of_constants * offset) +  A53]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B53]*STATES[(ORd_num_of_states * offset) + V])*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q53]))/10.0000))/( CONSTANTS[(ORd_num_of_constants * offset) +  A63]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B63]*STATES[(ORd_num_of_states * offset) + V])*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q63]))/10.0000)))*STATES[(ORd_num_of_states * offset) + IObound])+ (CONSTANTS[(ORd_num_of_constants * offset) +  Kt]/(1.00000+exp(- (STATES[(ORd_num_of_states * offset) + V] - CONSTANTS[(ORd_num_of_constants * offset) +  Vhalf])/6.78900)))*STATES[(ORd_num_of_states * offset) + Cbound]) -  CONSTANTS[(ORd_num_of_constants * offset) +  Kt]*STATES[(ORd_num_of_states * offset) + IObound];
RATES[(ORd_num_of_rates * offset) + Obound] = (( (( CONSTANTS[(ORd_num_of_constants * offset) +  Kmax]*CONSTANTS[(ORd_num_of_constants * offset) +  Ku]*pow( STATES[(ORd_num_of_states * offset) + D],CONSTANTS[(ORd_num_of_constants * offset) +  n]))/(pow( STATES[(ORd_num_of_states * offset) + D],CONSTANTS[(ORd_num_of_constants * offset) +  n])+CONSTANTS[(ORd_num_of_constants * offset) +  halfmax]))*STATES[(ORd_num_of_states * offset) + O] -  CONSTANTS[(ORd_num_of_constants * offset) +  Ku]*STATES[(ORd_num_of_states * offset) + Obound])+ (CONSTANTS[(ORd_num_of_constants * offset) +  Kt]/(1.00000+exp(- (STATES[(ORd_num_of_states * offset) + V] - CONSTANTS[(ORd_num_of_constants * offset) +  Vhalf])/6.78900)))*STATES[(ORd_num_of_states * offset) + Cbound]) -  CONSTANTS[(ORd_num_of_constants * offset) +  Kt]*STATES[(ORd_num_of_states * offset) + Obound];
RATES[(ORd_num_of_rates * offset) + Cbound] = - ( (CONSTANTS[(ORd_num_of_constants * offset) +  Kt]/(1.00000+exp(- (STATES[(ORd_num_of_states * offset) + V] - CONSTANTS[(ORd_num_of_constants * offset) +  Vhalf])/6.78900)))*STATES[(ORd_num_of_states * offset) + Cbound] -  CONSTANTS[(ORd_num_of_constants * offset) +  Kt]*STATES[(ORd_num_of_states * offset) + Obound]) - ( (CONSTANTS[(ORd_num_of_constants * offset) +  Kt]/(1.00000+exp(- (STATES[(ORd_num_of_states * offset) + V] - CONSTANTS[(ORd_num_of_constants * offset) +  Vhalf])/6.78900)))*STATES[(ORd_num_of_states * offset) + Cbound] -  CONSTANTS[(ORd_num_of_constants * offset) +  Kt]*STATES[(ORd_num_of_states * offset) + IObound]);
RATES[(ORd_num_of_rates * offset) + hL] = (ALGEBRAIC[(ORd_num_of_algebraic * offset) + hLss] - STATES[(ORd_num_of_states * offset) + hL])/CONSTANTS[(ORd_num_of_constants * offset) +  thL];
RATES[(ORd_num_of_rates * offset) + hLp] = (ALGEBRAIC[(ORd_num_of_algebraic * offset) + hLssp] - STATES[(ORd_num_of_states * offset) + hLp])/CONSTANTS[(ORd_num_of_constants * offset) +  thLp];
RATES[(ORd_num_of_rates * offset) + m] = (ALGEBRAIC[(ORd_num_of_algebraic * offset) + mss] - STATES[(ORd_num_of_states * offset) + m])/ALGEBRAIC[(ORd_num_of_algebraic * offset) + tm];
RATES[(ORd_num_of_rates * offset) + hf] = (ALGEBRAIC[(ORd_num_of_algebraic * offset) + hss] - STATES[(ORd_num_of_states * offset) + hf])/ALGEBRAIC[(ORd_num_of_algebraic * offset) + thf];
RATES[(ORd_num_of_rates * offset) + hs] = (ALGEBRAIC[(ORd_num_of_algebraic * offset) + hss] - STATES[(ORd_num_of_states * offset) + hs])/ALGEBRAIC[(ORd_num_of_algebraic * offset) + ths];
RATES[(ORd_num_of_rates * offset) + a] = (ALGEBRAIC[(ORd_num_of_algebraic * offset) + ass] - STATES[(ORd_num_of_states * offset) + a])/ALGEBRAIC[(ORd_num_of_algebraic * offset) + ta];
RATES[(ORd_num_of_rates * offset) + d] = (ALGEBRAIC[(ORd_num_of_algebraic * offset) + dss] - STATES[(ORd_num_of_states * offset) + d])/ALGEBRAIC[(ORd_num_of_algebraic * offset) + td];
RATES[(ORd_num_of_rates * offset) + ff] = (ALGEBRAIC[(ORd_num_of_algebraic * offset) + fss] - STATES[(ORd_num_of_states * offset) + ff])/ALGEBRAIC[(ORd_num_of_algebraic * offset) + tff];
RATES[(ORd_num_of_rates * offset) + fs] = (ALGEBRAIC[(ORd_num_of_algebraic * offset) + fss] - STATES[(ORd_num_of_states * offset) + fs])/ALGEBRAIC[(ORd_num_of_algebraic * offset) + tfs];
RATES[(ORd_num_of_rates * offset) + jca] = (ALGEBRAIC[(ORd_num_of_algebraic * offset) + fcass] - STATES[(ORd_num_of_states * offset) + jca])/CONSTANTS[(ORd_num_of_constants * offset) +  tjca];
RATES[(ORd_num_of_rates * offset) + nca] =  ALGEBRAIC[(ORd_num_of_algebraic * offset) + anca]*CONSTANTS[(ORd_num_of_constants * offset) +  k2n] -  STATES[(ORd_num_of_states * offset) + nca]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + km2n];
RATES[(ORd_num_of_rates * offset) + xs1] = (ALGEBRAIC[(ORd_num_of_algebraic * offset) + xs1ss] - STATES[(ORd_num_of_states * offset) + xs1])/ALGEBRAIC[(ORd_num_of_algebraic * offset) + txs1];
RATES[(ORd_num_of_rates * offset) + xk1] = (ALGEBRAIC[(ORd_num_of_algebraic * offset) + xk1ss] - STATES[(ORd_num_of_states * offset) + xk1])/ALGEBRAIC[(ORd_num_of_algebraic * offset) + txk1];
RATES[(ORd_num_of_rates * offset) + CaMKt] =  CONSTANTS[(ORd_num_of_constants * offset) +  aCaMK]*ALGEBRAIC[(ORd_num_of_algebraic * offset) + CaMKb]*(ALGEBRAIC[(ORd_num_of_algebraic * offset) + CaMKb]+STATES[(ORd_num_of_states * offset) + CaMKt]) -  CONSTANTS[(ORd_num_of_constants * offset) +  bCaMK]*STATES[(ORd_num_of_states * offset) + CaMKt];
RATES[(ORd_num_of_rates * offset) + j] = (ALGEBRAIC[(ORd_num_of_algebraic * offset) + jss] - STATES[(ORd_num_of_states * offset) + j])/ALGEBRAIC[(ORd_num_of_algebraic * offset) + tj];
RATES[(ORd_num_of_rates * offset) + ap] = (ALGEBRAIC[(ORd_num_of_algebraic * offset) + assp] - STATES[(ORd_num_of_states * offset) + ap])/ALGEBRAIC[(ORd_num_of_algebraic * offset) + ta];
RATES[(ORd_num_of_rates * offset) + fcaf] = (ALGEBRAIC[(ORd_num_of_algebraic * offset) + fcass] - STATES[(ORd_num_of_states * offset) + fcaf])/ALGEBRAIC[(ORd_num_of_algebraic * offset) + tfcaf];
RATES[(ORd_num_of_rates * offset) + fcas] = (ALGEBRAIC[(ORd_num_of_algebraic * offset) + fcass] - STATES[(ORd_num_of_states * offset) + fcas])/ALGEBRAIC[(ORd_num_of_algebraic * offset) + tfcas];
RATES[(ORd_num_of_rates * offset) + ffp] = (ALGEBRAIC[(ORd_num_of_algebraic * offset) + fss] - STATES[(ORd_num_of_states * offset) + ffp])/ALGEBRAIC[(ORd_num_of_algebraic * offset) + tffp];
RATES[(ORd_num_of_rates * offset) + xs2] = (ALGEBRAIC[(ORd_num_of_algebraic * offset) + xs2ss] - STATES[(ORd_num_of_states * offset) + xs2])/ALGEBRAIC[(ORd_num_of_algebraic * offset) + txs2];
RATES[(ORd_num_of_rates * offset) + hsp] = (ALGEBRAIC[(ORd_num_of_algebraic * offset) + hssp] - STATES[(ORd_num_of_states * offset) + hsp])/ALGEBRAIC[(ORd_num_of_algebraic * offset) + thsp];
RATES[(ORd_num_of_rates * offset) + jp] = (ALGEBRAIC[(ORd_num_of_algebraic * offset) + jss] - STATES[(ORd_num_of_states * offset) + jp])/ALGEBRAIC[(ORd_num_of_algebraic * offset) + tjp];
RATES[(ORd_num_of_rates * offset) + mL] = (ALGEBRAIC[(ORd_num_of_algebraic * offset) + mLss] - STATES[(ORd_num_of_states * offset) + mL])/ALGEBRAIC[(ORd_num_of_algebraic * offset) + tmL];
RATES[(ORd_num_of_rates * offset) + fcafp] = (ALGEBRAIC[(ORd_num_of_algebraic * offset) + fcass] - STATES[(ORd_num_of_states * offset) + fcafp])/ALGEBRAIC[(ORd_num_of_algebraic * offset) + tfcafp];
RATES[(ORd_num_of_rates * offset) + iF] = (ALGEBRAIC[(ORd_num_of_algebraic * offset) + iss] - STATES[(ORd_num_of_states * offset) + iF])/ALGEBRAIC[(ORd_num_of_algebraic * offset) + tiF];
RATES[(ORd_num_of_rates * offset) + iS] = (ALGEBRAIC[(ORd_num_of_algebraic * offset) + iss] - STATES[(ORd_num_of_states * offset) + iS])/ALGEBRAIC[(ORd_num_of_algebraic * offset) + tiS];
RATES[(ORd_num_of_rates * offset) + iFp] = (ALGEBRAIC[(ORd_num_of_algebraic * offset) + iss] - STATES[(ORd_num_of_states * offset) + iFp])/ALGEBRAIC[(ORd_num_of_algebraic * offset) + tiFp];
RATES[(ORd_num_of_rates * offset) + iSp] = (ALGEBRAIC[(ORd_num_of_algebraic * offset) + iss] - STATES[(ORd_num_of_states * offset) + iSp])/ALGEBRAIC[(ORd_num_of_algebraic * offset) + tiSp];
RATES[(ORd_num_of_rates * offset) + Jrelnp] = (ALGEBRAIC[(ORd_num_of_algebraic * offset) + Jrel_inf] - STATES[(ORd_num_of_states * offset) + Jrelnp])/ALGEBRAIC[(ORd_num_of_algebraic * offset) + tau_rel];
RATES[(ORd_num_of_rates * offset) + Jrelp] = (ALGEBRAIC[(ORd_num_of_algebraic * offset) + Jrel_infp] - STATES[(ORd_num_of_states * offset) + Jrelp])/ALGEBRAIC[(ORd_num_of_algebraic * offset) + tau_relp];
RATES[(ORd_num_of_rates * offset) + ki] = ( - ((ALGEBRAIC[(ORd_num_of_algebraic * offset) + Ito]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + IKr]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + IKs]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + IK1]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + IKb]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + Istim]) -  2.00000*ALGEBRAIC[(ORd_num_of_algebraic * offset) + INaK])*CONSTANTS[(ORd_num_of_constants * offset) +  cm]*CONSTANTS[(ORd_num_of_constants * offset) +  Acap])/( CONSTANTS[(ORd_num_of_constants * offset) +  F]*CONSTANTS[(ORd_num_of_constants * offset) +  vmyo])+( ALGEBRAIC[(ORd_num_of_algebraic * offset) + JdiffK]*CONSTANTS[(ORd_num_of_constants * offset) +  vss])/CONSTANTS[(ORd_num_of_constants * offset) +  vmyo];
RATES[(ORd_num_of_rates * offset) + kss] = ( - ALGEBRAIC[(ORd_num_of_algebraic * offset) + ICaK]*CONSTANTS[(ORd_num_of_constants * offset) +  cm]*CONSTANTS[(ORd_num_of_constants * offset) +  Acap])/( CONSTANTS[(ORd_num_of_constants * offset) +  F]*CONSTANTS[(ORd_num_of_constants * offset) +  vss]) - ALGEBRAIC[(ORd_num_of_algebraic * offset) + JdiffK];
RATES[(ORd_num_of_rates * offset) + nai] = ( - (ALGEBRAIC[(ORd_num_of_algebraic * offset) + INa]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + INaL]+ 3.00000*ALGEBRAIC[(ORd_num_of_algebraic * offset) + INaCa_i]+ 3.00000*ALGEBRAIC[(ORd_num_of_algebraic * offset) + INaK]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + INab])*CONSTANTS[(ORd_num_of_constants * offset) +  Acap]*CONSTANTS[(ORd_num_of_constants * offset) +  cm])/( CONSTANTS[(ORd_num_of_constants * offset) +  F]*CONSTANTS[(ORd_num_of_constants * offset) +  vmyo])+( ALGEBRAIC[(ORd_num_of_algebraic * offset) + JdiffNa]*CONSTANTS[(ORd_num_of_constants * offset) +  vss])/CONSTANTS[(ORd_num_of_constants * offset) +  vmyo];
RATES[(ORd_num_of_rates * offset) + nass] = ( - (ALGEBRAIC[(ORd_num_of_algebraic * offset) + ICaNa]+ 3.00000*ALGEBRAIC[(ORd_num_of_algebraic * offset) + INaCa_ss])*CONSTANTS[(ORd_num_of_constants * offset) +  cm]*CONSTANTS[(ORd_num_of_constants * offset) +  Acap])/( CONSTANTS[(ORd_num_of_constants * offset) +  F]*CONSTANTS[(ORd_num_of_constants * offset) +  vss]) - ALGEBRAIC[(ORd_num_of_algebraic * offset) + JdiffNa];
RATES[(ORd_num_of_rates * offset) + V] = - (ALGEBRAIC[(ORd_num_of_algebraic * offset) + INa]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + INaL]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + Ito]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + ICaL]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + ICaNa]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + ICaK]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + IKr]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + IKs]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + IK1]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + INaCa_i]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + INaCa_ss]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + INaK]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + INab]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + IKb]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + IpCa]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + ICab]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + Istim]);
RATES[(ORd_num_of_rates * offset) + cass] =  ALGEBRAIC[(ORd_num_of_algebraic * offset) + Bcass]*((( - (ALGEBRAIC[(ORd_num_of_algebraic * offset) + ICaL] -  2.00000*ALGEBRAIC[(ORd_num_of_algebraic * offset) + INaCa_ss])*CONSTANTS[(ORd_num_of_constants * offset) +  cm]*CONSTANTS[(ORd_num_of_constants * offset) +  Acap])/( 2.00000*CONSTANTS[(ORd_num_of_constants * offset) +  F]*CONSTANTS[(ORd_num_of_constants * offset) +  vss])+( ALGEBRAIC[(ORd_num_of_algebraic * offset) + Jrel]*CONSTANTS[(ORd_num_of_constants * offset) +  vjsr])/CONSTANTS[(ORd_num_of_constants * offset) +  vss]) - ALGEBRAIC[(ORd_num_of_algebraic * offset) + Jdiff]);
//modified for coupling
RATES[(offset * ORd_num_of_rates) + ca_trpn] = CONSTANTS[(offset * ORd_num_of_constants) + trpnmax] * land_trpn;
// if (offset == 1) printf("in ord: ca_trpn: %lf is %lf times %lf\n\n", RATES[(offset * ORd_num_of_rates) + ca_trpn], CONSTANTS[(offset * ORd_num_of_constants) + trpnmax], land_trpn );
RATES[(ORd_num_of_rates * offset) + cai] =  ALGEBRAIC[(ORd_num_of_algebraic * offset) + Bcai]*((( - ((ALGEBRAIC[(ORd_num_of_algebraic * offset) + IpCa]+ALGEBRAIC[(ORd_num_of_algebraic * offset) + ICab]) -  2.00000*ALGEBRAIC[(ORd_num_of_algebraic * offset) + INaCa_i])*CONSTANTS[(ORd_num_of_constants * offset) +  cm]*CONSTANTS[(ORd_num_of_constants * offset) +  Acap])/( 2.00000*CONSTANTS[(ORd_num_of_constants * offset) +  F]*CONSTANTS[(ORd_num_of_constants * offset) +  vmyo]) - ( ALGEBRAIC[(ORd_num_of_algebraic * offset) + Jup]*CONSTANTS[(ORd_num_of_constants * offset) +  vnsr])/CONSTANTS[(ORd_num_of_constants * offset) +  vmyo])+( ALGEBRAIC[(ORd_num_of_algebraic * offset) + Jdiff]*CONSTANTS[(ORd_num_of_constants * offset) +  vss])/CONSTANTS[(ORd_num_of_constants * offset) +  vmyo] - RATES[(offset * ORd_num_of_rates) + ca_trpn]); //modified
RATES[(ORd_num_of_rates * offset) + cansr] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + Jup] - ( ALGEBRAIC[(ORd_num_of_algebraic * offset) + Jtr]*CONSTANTS[(ORd_num_of_constants * offset) +  vjsr])/CONSTANTS[(ORd_num_of_constants * offset) +  vnsr];
RATES[(ORd_num_of_rates * offset) + cajsr] =  ALGEBRAIC[(ORd_num_of_algebraic * offset) + Bcajsr]*(ALGEBRAIC[(ORd_num_of_algebraic * offset) + Jtr] - ALGEBRAIC[(ORd_num_of_algebraic * offset) + Jrel]);
}


__device__ void solveAnalytical(double *CONSTANTS, double *STATES, double *ALGEBRAIC, double *RATES, double dt, int offset)
{
////==============
////Exact solution
////==============
////INa
  STATES[(ORd_num_of_states * offset) + m] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + mss] - (ALGEBRAIC[(ORd_num_of_algebraic * offset) + mss] - STATES[(ORd_num_of_states * offset) + m]) * exp(-dt / ALGEBRAIC[(ORd_num_of_algebraic * offset) + tm]);
  STATES[(ORd_num_of_states * offset) + hf] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + hss] - (ALGEBRAIC[(ORd_num_of_algebraic * offset) + hss] - STATES[(ORd_num_of_states * offset) + hf]) * exp(-dt / ALGEBRAIC[(ORd_num_of_algebraic * offset) + thf]);
  STATES[(ORd_num_of_states * offset) + hs] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + hss] - (ALGEBRAIC[(ORd_num_of_algebraic * offset) + hss] - STATES[(ORd_num_of_states * offset) + hs]) * exp(-dt / ALGEBRAIC[(ORd_num_of_algebraic * offset) + ths]);
  STATES[(ORd_num_of_states * offset) + j] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + jss] - (ALGEBRAIC[(ORd_num_of_algebraic * offset) + jss] - STATES[(ORd_num_of_states * offset) + j]) * exp(-dt / ALGEBRAIC[(ORd_num_of_algebraic * offset) + tj]);
  STATES[(ORd_num_of_states * offset) + hsp] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + hssp] - (ALGEBRAIC[(ORd_num_of_algebraic * offset) + hssp] - STATES[(ORd_num_of_states * offset) + hsp]) * exp(-dt / ALGEBRAIC[(ORd_num_of_algebraic * offset) + thsp]);
  STATES[(ORd_num_of_states * offset) + jp] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + jss] - (ALGEBRAIC[(ORd_num_of_algebraic * offset) + jss] - STATES[(ORd_num_of_states * offset) + jp]) * exp(-dt / ALGEBRAIC[(ORd_num_of_algebraic * offset) + tjp]);
  STATES[(ORd_num_of_states * offset) + mL] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + mLss] - (ALGEBRAIC[(ORd_num_of_algebraic * offset) + mLss] - STATES[(ORd_num_of_states * offset) + mL]) * exp(-dt / ALGEBRAIC[(ORd_num_of_algebraic * offset) + tmL]);
  STATES[(ORd_num_of_states * offset) + hL] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + hLss] - (ALGEBRAIC[(ORd_num_of_algebraic * offset) + hLss] - STATES[(ORd_num_of_states * offset) + hL]) * exp(-dt / CONSTANTS[(ORd_num_of_constants * offset) +  thL]);
  STATES[(ORd_num_of_states * offset) + hLp] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + hLssp] - (ALGEBRAIC[(ORd_num_of_algebraic * offset) + hLssp] - STATES[(ORd_num_of_states * offset) + hLp]) * exp(-dt / CONSTANTS[(ORd_num_of_constants * offset) +  thLp]);
////Ito
  STATES[(ORd_num_of_states * offset) + a] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + ass] - (ALGEBRAIC[(ORd_num_of_algebraic * offset) + ass] - STATES[(ORd_num_of_states * offset) + a]) * exp(-dt / ALGEBRAIC[(ORd_num_of_algebraic * offset) + ta]);
  STATES[(ORd_num_of_states * offset) + iF] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + iss] - (ALGEBRAIC[(ORd_num_of_algebraic * offset) + iss] - STATES[(ORd_num_of_states * offset) + iF]) * exp(-dt / ALGEBRAIC[(ORd_num_of_algebraic * offset) + tiF]);
  STATES[(ORd_num_of_states * offset) + iS] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + iss] - (ALGEBRAIC[(ORd_num_of_algebraic * offset) + iss] - STATES[(ORd_num_of_states * offset) + iS]) * exp(-dt / ALGEBRAIC[(ORd_num_of_algebraic * offset) + tiS]);
  STATES[(ORd_num_of_states * offset) + ap] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + assp] - (ALGEBRAIC[(ORd_num_of_algebraic * offset) + assp] - STATES[(ORd_num_of_states * offset) + ap]) * exp(-dt / ALGEBRAIC[(ORd_num_of_algebraic * offset) + ta]);
  STATES[(ORd_num_of_states * offset) + iFp] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + iss] - (ALGEBRAIC[(ORd_num_of_algebraic * offset) + iss] - STATES[(ORd_num_of_states * offset) + iFp]) * exp(-dt / ALGEBRAIC[(ORd_num_of_algebraic * offset) + tiFp]);
  STATES[(ORd_num_of_states * offset) + iSp] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + iss] - (ALGEBRAIC[(ORd_num_of_algebraic * offset) + iss] - STATES[(ORd_num_of_states * offset) + iSp]) * exp(-dt / ALGEBRAIC[(ORd_num_of_algebraic * offset) + tiSp]);
////ICaL
  STATES[(ORd_num_of_states * offset) + d] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + dss] - (ALGEBRAIC[(ORd_num_of_algebraic * offset) + dss] - STATES[(ORd_num_of_states * offset) + d]) * exp(-dt / ALGEBRAIC[(ORd_num_of_algebraic * offset) + td]);
  STATES[(ORd_num_of_states * offset) + ff] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + fss] - (ALGEBRAIC[(ORd_num_of_algebraic * offset) + fss] - STATES[(ORd_num_of_states * offset) + ff]) * exp(-dt / ALGEBRAIC[(ORd_num_of_algebraic * offset) + tff]);
  STATES[(ORd_num_of_states * offset) + fs] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + fss] - (ALGEBRAIC[(ORd_num_of_algebraic * offset) + fss] - STATES[(ORd_num_of_states * offset) + fs]) * exp(-dt / ALGEBRAIC[(ORd_num_of_algebraic * offset) + tfs]);
  STATES[(ORd_num_of_states * offset) + fcaf] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + fcass] - (ALGEBRAIC[(ORd_num_of_algebraic * offset) + fcass] - STATES[(ORd_num_of_states * offset) + fcaf]) * exp(-dt / ALGEBRAIC[(ORd_num_of_algebraic * offset) + tfcaf]);
  STATES[(ORd_num_of_states * offset) + fcas] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + fcass] - (ALGEBRAIC[(ORd_num_of_algebraic * offset) + fcass] - STATES[(ORd_num_of_states * offset) + fcas]) * exp(-dt / ALGEBRAIC[(ORd_num_of_algebraic * offset) + tfcas]);
  STATES[(ORd_num_of_states * offset) + jca] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + fcass] - (ALGEBRAIC[(ORd_num_of_algebraic * offset) + fcass] - STATES[(ORd_num_of_states * offset) + jca]) * exp(- dt / CONSTANTS[(ORd_num_of_constants * offset) +  tjca]);
  STATES[(ORd_num_of_states * offset) + ffp] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + fss] - (ALGEBRAIC[(ORd_num_of_algebraic * offset) + fss] - STATES[(ORd_num_of_states * offset) + ffp]) * exp(-dt / ALGEBRAIC[(ORd_num_of_algebraic * offset) + tffp]);
  STATES[(ORd_num_of_states * offset) + fcafp] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + fcass] - (ALGEBRAIC[(ORd_num_of_algebraic * offset) + fcass] - STATES[(ORd_num_of_states * offset) + fcafp]) * exp(-d / ALGEBRAIC[(ORd_num_of_algebraic * offset) + tfcafp]);
  STATES[(ORd_num_of_states * offset) + nca] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + anca] * CONSTANTS[(ORd_num_of_constants * offset) +  k2n] / ALGEBRAIC[(ORd_num_of_algebraic * offset) + km2n] -
      (ALGEBRAIC[(ORd_num_of_algebraic * offset) + anca] * CONSTANTS[(ORd_num_of_constants * offset) +  k2n] / ALGEBRAIC[(ORd_num_of_algebraic * offset) + km2n] - STATES[(ORd_num_of_states * offset) + nca]) * exp(-ALGEBRAIC[(ORd_num_of_algebraic * offset) + km2n] * dt);
////IKs
  STATES[(ORd_num_of_states * offset) + xs1] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + xs1ss] - (ALGEBRAIC[(ORd_num_of_algebraic * offset) + xs1ss] - STATES[(ORd_num_of_states * offset) + xs1]) * exp(-dt / ALGEBRAIC[(ORd_num_of_algebraic * offset) + txs1]);
  STATES[(ORd_num_of_states * offset) + xs2] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + xs2ss] - (ALGEBRAIC[(ORd_num_of_algebraic * offset) + xs2ss] - STATES[(ORd_num_of_states * offset) + xs2]) * exp(-dt / ALGEBRAIC[(ORd_num_of_algebraic * offset) + txs2]);
////IK1
  STATES[(ORd_num_of_states * offset) + xk1] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + xk1ss] - (ALGEBRAIC[(ORd_num_of_algebraic * offset) + xk1ss] - STATES[(ORd_num_of_states * offset) + xk1]) * exp(-dt / ALGEBRAIC[(ORd_num_of_algebraic * offset) + txk1]);
////RyR receptors
  STATES[(ORd_num_of_states * offset) + Jrelnp] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + Jrel_inf] - (ALGEBRAIC[(ORd_num_of_algebraic * offset) + Jrel_inf] - STATES[(ORd_num_of_states * offset) + Jrelnp]) * exp(-dt / ALGEBRAIC[(ORd_num_of_algebraic * offset) + tau_rel]);
  STATES[(ORd_num_of_states * offset) + Jrelp] = ALGEBRAIC[(ORd_num_of_algebraic * offset) + Jrel_infp] - (ALGEBRAIC[(ORd_num_of_algebraic * offset) + Jrel_infp] - STATES[(ORd_num_of_states * offset) + Jrelp]) * exp(-dt / ALGEBRAIC[(ORd_num_of_algebraic * offset) + tau_relp]);
////=============================
////Approximated solution (Backward Euler)
////=============================
////IKr
  double* coeffs = new double[31];
  coeffs[0] = - CONSTANTS[(ORd_num_of_constants * offset) +  A11]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B11]*STATES[(ORd_num_of_states * offset) + V])*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q11]))/10.0000) - CONSTANTS[(ORd_num_of_constants * offset) +  A61]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B61]*STATES[(ORd_num_of_states * offset) + V])*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q61]))/10.0000);
  coeffs[1] = CONSTANTS[(ORd_num_of_constants * offset) +  A21]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B21]*STATES[(ORd_num_of_states * offset) + V])*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q21]))/10.0000);
  coeffs[2] = CONSTANTS[(ORd_num_of_constants * offset) +  A51]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B51]*STATES[(ORd_num_of_states * offset) + V])*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q51]))/10.0000);

  coeffs[3] = CONSTANTS[(ORd_num_of_constants * offset) +  A11]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B11]*STATES[(ORd_num_of_states * offset) + V])*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q11]))/10.0000);
  coeffs[4] = - CONSTANTS[(ORd_num_of_constants * offset) +  A21]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B21]*STATES[(ORd_num_of_states * offset) + V])*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q21]))/10.0000) - CONSTANTS[(ORd_num_of_constants * offset) +  A3]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B3]*STATES[(ORd_num_of_states * offset) + V])*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q3]))/10.0000) - CONSTANTS[(ORd_num_of_constants * offset) +  A62]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B62]*STATES[(ORd_num_of_states * offset) + V])*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q62]))/10.0000);
  coeffs[5] = CONSTANTS[(ORd_num_of_constants * offset) +  A52]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B52]*STATES[(ORd_num_of_states * offset) + V])*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q52]))/10.0000);
  coeffs[6] = CONSTANTS[(ORd_num_of_constants * offset) +  A4]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B4]*STATES[(ORd_num_of_states * offset) + V])*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q4]))/10.0000);

  coeffs[7] = CONSTANTS[(ORd_num_of_constants * offset) +  A61]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B61]*STATES[(ORd_num_of_states * offset) + V])*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q61]))/10.0000);
  coeffs[8] = - CONSTANTS[(ORd_num_of_constants * offset) +  A1]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B1]*STATES[(ORd_num_of_states * offset) + V])*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q1]))/10.0000) - CONSTANTS[(ORd_num_of_constants * offset) +  A51]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B51]*STATES[(ORd_num_of_states * offset) + V])*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q51]))/10.0000);
  coeffs[9] = CONSTANTS[(ORd_num_of_constants * offset) +  A2]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B2]*STATES[(ORd_num_of_states * offset) + V])*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q2]))/10.0000);

  coeffs[10] = CONSTANTS[(ORd_num_of_constants * offset) +  A62]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B62]*STATES[(ORd_num_of_states * offset) + V])*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q62]))/10.0000);
  coeffs[11] = CONSTANTS[(ORd_num_of_constants * offset) +  A1]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B1]*STATES[(ORd_num_of_states * offset) + V])*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q1]))/10.0000);
  coeffs[12] = - CONSTANTS[(ORd_num_of_constants * offset) +  A2]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B2]*STATES[(ORd_num_of_states * offset) + V])*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q2]))/10.0000) - CONSTANTS[(ORd_num_of_constants * offset) +  A31]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B31]*STATES[(ORd_num_of_states * offset) + V])*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q31]))/10.0000) - CONSTANTS[(ORd_num_of_constants * offset) +  A52]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B52]*STATES[(ORd_num_of_states * offset) + V])*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q52]))/10.0000);
  coeffs[13] = CONSTANTS[(ORd_num_of_constants * offset) +  A41]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B41]*STATES[(ORd_num_of_states * offset) + V])*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q41]))/10.0000);

  coeffs[14] = CONSTANTS[(ORd_num_of_constants * offset) +  A31]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B31]*STATES[(ORd_num_of_states * offset) + V])*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q31]))/10.0000);
  coeffs[15] = - CONSTANTS[(ORd_num_of_constants * offset) +  A41]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B41]*STATES[(ORd_num_of_states * offset) + V])*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q41]))/10.0000) - CONSTANTS[(ORd_num_of_constants * offset) +  A53]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B53]*STATES[(ORd_num_of_states * offset) + V])*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q53]))/10.0000) - (( CONSTANTS[(ORd_num_of_constants * offset) +  Kmax]*CONSTANTS[(ORd_num_of_constants * offset) +  Ku]*pow( STATES[(ORd_num_of_states * offset) + D],CONSTANTS[(ORd_num_of_constants * offset) +  n]))/(pow( STATES[(ORd_num_of_states * offset) + D],CONSTANTS[(ORd_num_of_constants * offset) +  n])+CONSTANTS[(ORd_num_of_constants * offset) +  halfmax]));
  coeffs[16] = CONSTANTS[(ORd_num_of_constants * offset) +  A63]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B63]*STATES[(ORd_num_of_states * offset) + V])*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q63]))/10.0000);
  coeffs[17] = CONSTANTS[(ORd_num_of_constants * offset) +  Kt];

  coeffs[18] = CONSTANTS[(ORd_num_of_constants * offset) +  A3]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B3]*STATES[(ORd_num_of_states * offset) + V])*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q3]))/10.0000);
  coeffs[19] = CONSTANTS[(ORd_num_of_constants * offset) +  A53]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B53]*STATES[(ORd_num_of_states * offset) + V])*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q53]))/10.0000);
  coeffs[20] = - CONSTANTS[(ORd_num_of_constants * offset) +  A4]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B4]*STATES[(ORd_num_of_states * offset) + V])*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q4]))/10.0000) - CONSTANTS[(ORd_num_of_constants * offset) +  A63]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B63]*STATES[(ORd_num_of_states * offset) + V])*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q63]))/10.0000) - (( CONSTANTS[(ORd_num_of_constants * offset) +  Kmax]*CONSTANTS[(ORd_num_of_constants * offset) +  Ku]*pow( STATES[(ORd_num_of_states * offset) + D],CONSTANTS[(ORd_num_of_constants * offset) +  n]))/(pow( STATES[(ORd_num_of_states * offset) + D],CONSTANTS[(ORd_num_of_constants * offset) +  n])+CONSTANTS[(ORd_num_of_constants * offset) +  halfmax]));
  coeffs[21] = (( CONSTANTS[(ORd_num_of_constants * offset) +  Ku]*CONSTANTS[(ORd_num_of_constants * offset) +  A53]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B53]*STATES[(ORd_num_of_states * offset) + V])*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q53]))/10.0000))/( CONSTANTS[(ORd_num_of_constants * offset) +  A63]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B63]*STATES[(ORd_num_of_states * offset) + V])*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q63]))/10.0000)));

  coeffs[22] = (( CONSTANTS[(ORd_num_of_constants * offset) +  Kmax]*CONSTANTS[(ORd_num_of_constants * offset) +  Ku]*pow( STATES[(ORd_num_of_states * offset) + D],CONSTANTS[(ORd_num_of_constants * offset) +  n]))/(pow( STATES[(ORd_num_of_states * offset) + D],CONSTANTS[(ORd_num_of_constants * offset) +  n])+CONSTANTS[(ORd_num_of_constants * offset) +  halfmax]));
  coeffs[23] = -  CONSTANTS[(ORd_num_of_constants * offset) +  Ku] - CONSTANTS[(ORd_num_of_constants * offset) +  Kt];
  coeffs[24] = (CONSTANTS[(ORd_num_of_constants * offset) +  Kt]/(1.00000+exp(- (STATES[(ORd_num_of_states * offset) + V] - CONSTANTS[(ORd_num_of_constants * offset) +  Vhalf])/6.78900)));

  coeffs[25] = (( CONSTANTS[(ORd_num_of_constants * offset) +  Kmax]*CONSTANTS[(ORd_num_of_constants * offset) +  Ku]*pow( STATES[(ORd_num_of_states * offset) + D],CONSTANTS[(ORd_num_of_constants * offset) +  n]))/(pow( STATES[(ORd_num_of_states * offset) + D],CONSTANTS[(ORd_num_of_constants * offset) +  n])+CONSTANTS[(ORd_num_of_constants * offset) +  halfmax]));
  coeffs[26] = - (( CONSTANTS[(ORd_num_of_constants * offset) +  Ku]*CONSTANTS[(ORd_num_of_constants * offset) +  A53]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B53]*STATES[(ORd_num_of_states * offset) + V])*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q53]))/10.0000))/( CONSTANTS[(ORd_num_of_constants * offset) +  A63]*exp( CONSTANTS[(ORd_num_of_constants * offset) +  B63]*STATES[(ORd_num_of_states * offset) + V])*exp(( (CONSTANTS[(ORd_num_of_constants * offset) +  Temp] - 20.0000)*log(CONSTANTS[(ORd_num_of_constants * offset) +  q63]))/10.0000))) - CONSTANTS[(ORd_num_of_constants * offset) +  Kt];
  coeffs[27] = (CONSTANTS[(ORd_num_of_constants * offset) +  Kt]/(1.00000+exp(- (STATES[(ORd_num_of_states * offset) + V] - CONSTANTS[(ORd_num_of_constants * offset) +  Vhalf])/6.78900)));

  coeffs[28] = CONSTANTS[(ORd_num_of_constants * offset) +  Kt];
  coeffs[29] = CONSTANTS[(ORd_num_of_constants * offset) +  Kt];
  coeffs[30] = - (CONSTANTS[(ORd_num_of_constants * offset) +  Kt]/(1.00000+exp(- (STATES[(ORd_num_of_states * offset) + V] - CONSTANTS[(ORd_num_of_constants * offset) +  Vhalf])/6.78900))) - (CONSTANTS[(ORd_num_of_constants * offset) +  Kt]/(1.00000+exp(- (STATES[(ORd_num_of_states * offset) + V] - CONSTANTS[(ORd_num_of_constants * offset) +  Vhalf])/6.78900)));
  int m = 9;
  double* a = new double[m*m]; // Flattened a
  a[0 * m + 0] = 1.0 - dt * coeffs[0];   a[0 * m + 1] = - dt * coeffs[1];     a[0 * m + 2] = - dt * coeffs[2];     a[0 * m + 3] = 0.0;                      a[0 * m + 4] = 0.0;                      a[0 * m + 5] = 0.0;                      a[0 * m + 6] = 0.0;                      a[0 * m + 7] = 0.0;                      a[0 * m + 8] = 0.0;
  a[1 * m + 0] = - dt * coeffs[3];       a[1 * m + 1] = 1.0 - dt * coeffs[4]; a[1 * m + 2] = 0.0;                  a[1 * m + 3] = - dt * coeffs[5];         a[1 * m + 4] = 0.0;                      a[1 * m + 5] = - dt * coeffs[6];         a[1 * m + 6] = 0.0;                      a[1 * m + 7] = 0.0;                      a[1 * m + 8] = 0.0;
  a[2 * m + 0] = - dt * coeffs[7];       a[2 * m + 1] = 0.0;                  a[2 * m + 2] = 1.0 - dt * coeffs[8]; a[2 * m + 3] = - dt * coeffs[9];         a[2 * m + 4] = 0.0;                      a[2 * m + 5] = 0.0;                      a[2 * m + 6] = 0.0;                      a[2 * m + 7] = 0.0;                      a[2 * m + 8] = 0.0;
  a[3 * m + 0] = 0.0;                    a[3 * m + 1] = - dt * coeffs[10];    a[3 * m + 2] = - dt * coeffs[11];    a[3 * m + 3] = 1.0 - dt * coeffs[12];    a[3 * m + 4] = - dt * coeffs[13];        a[3 * m + 5] = 0.0;                      a[3 * m + 6] = 0.0;                      a[3 * m + 7] = 0.0;                      a[3 * m + 8] = 0.0;
  a[4 * m + 0] = 0.0;                    a[4 * m + 1] = 0.0;                  a[4 * m + 2] = 0.0;                  a[4 * m + 3] = - dt * coeffs[14];        a[4 * m + 4] = 1.0 - dt * coeffs[15];    a[4 * m + 5] = - dt * coeffs[16];        a[4 * m + 6] = - dt * coeffs[17];        a[4 * m + 7] = 0.0;                      a[4 * m + 8] = 0.0;
  a[5 * m + 0] = 0.0;                    a[5 * m + 1] = - dt * coeffs[18];    a[5 * m + 2] = 0.0;                  a[5 * m + 3] = 0.0;                      a[5 * m + 4] = - dt * coeffs[19];        a[5 * m + 5] = 1.0 - dt * coeffs[20];    a[5 * m + 6] = - dt * coeffs[21];        a[5 * m + 7] = 0.0;                      a[5 * m + 8] = 0.0;
  a[6 * m + 0] = 0.0;                    a[6 * m + 1] = 0.0;                  a[6 * m + 2] = 0.0;                  a[6 * m + 3] = 0.0;                      a[6 * m + 4] = - dt * coeffs[22];        a[6 * m + 5] = 0.0;                      a[6 * m + 6] = 1.0 - dt * coeffs[23];    a[6 * m + 7] = 0.0;                      a[6 * m + 8] = - dt * coeffs[24];
  a[7 * m + 0] = 0.0;                    a[7 * m + 1] = 0.0;                  a[7 * m + 2] = 0.0;                  a[7 * m + 3] = 0.0;                      a[7 * m + 4] = 0.0;                      a[7 * m + 5] = - dt * coeffs[25];        a[7 * m + 6] = 0.0;                      a[7 * m + 7] = 1.0 - dt * coeffs[26];    a[7 * m + 8] = - dt * coeffs[27];
  a[8 * m + 0] = 0.0;                    a[8 * m + 1] = 0.0;                  a[8 * m + 2] = 0.0;                  a[8 * m + 3] = 0.0;                      a[8 * m + 4] = 0.0;                      a[8 * m + 5] = 0.0;                      a[8 * m + 6] = - dt * coeffs[28];        a[8 * m + 7] = - dt * coeffs[29];        a[8 * m + 8] = 1.0 - dt * coeffs[30];
  double* b = new double[m];
  b[0] = STATES[(ORd_num_of_states * offset) + IC1];
  b[1] = STATES[(ORd_num_of_states * offset) + IC2];
  b[2] = STATES[(ORd_num_of_states * offset) + C1];
  b[3] = STATES[(ORd_num_of_states * offset) + C2];
  b[4] = STATES[(ORd_num_of_states * offset) + O];
  b[5] = STATES[(ORd_num_of_states * offset) + IO];
  b[6] = STATES[(ORd_num_of_states * offset) + Obound];
  b[7] = STATES[(ORd_num_of_states * offset) + IObound];
  b[8] = STATES[(ORd_num_of_states * offset) + Cbound];
  double* x = new double[m];
  for(int i = 0; i < m; i++){
    x[i] = 0.0;
  }
  ___gaussElimination(a,b,x,m); // gpu capable?
  STATES[(ORd_num_of_states * offset) + IC1] = x[0];
  STATES[(ORd_num_of_states * offset) + IC2] = x[1];
  STATES[(ORd_num_of_states * offset) + C1] = x[2];
  STATES[(ORd_num_of_states * offset) + C2] = x[3];
  STATES[(ORd_num_of_states * offset) + O] = x[4];
  STATES[(ORd_num_of_states * offset) + IO] = x[5];
  STATES[(ORd_num_of_states * offset) + Obound] = x[6];
  STATES[(ORd_num_of_states * offset) + IObound] = x[7];
  STATES[(ORd_num_of_states * offset) + Cbound] = x[8];
  delete[] coeffs;
  delete[] a;
  delete[] b;
  delete[] x;
//  STATES[IC1] = STATES[IC1] + RATES[IC1] * dt;
//  STATES[IC2] = STATES[IC2] + RATES[IC2] * dt;
//  STATES[C1] = STATES[C1] + RATES[C1] * dt;
//  STATES[C2] = STATES[C2] + RATES[C2] * dt;
//  STATES[O] = STATES[O] + RATES[O] * dt;
//  STATES[IO] = STATES[IO] + RATES[IO] * dt;
//  STATES[D] = STATES[D] + RATES[D] * dt;
//  STATES[IObound] = STATES[IObound] + RATES[IObound] * dt;
//  STATES[Obound] = STATES[Obound] + RATES[Obound] * dt;
//  STATES[Cbound] = STATES[Cbound] + RATES[Cbound] * dt;
////=============================
////Approximated solution (Forward Euler)
////=============================
////CaMK
  STATES[(ORd_num_of_states * offset) + CaMKt] = STATES[(ORd_num_of_states * offset) + CaMKt] + RATES[(ORd_num_of_rates * offset) + CaMKt] * dt;
////Membrane potential
  STATES[(ORd_num_of_states * offset) + V] = STATES[(ORd_num_of_states * offset) + V] + RATES[(ORd_num_of_rates * offset) + V] * dt;
////Ion Concentrations and Buffers
  STATES[(ORd_num_of_states * offset) + nai] = STATES[(ORd_num_of_states * offset) + nai] + RATES[(ORd_num_of_rates * offset) + nai] * dt;
  STATES[(ORd_num_of_states * offset) + nass] = STATES[(ORd_num_of_states * offset) + nass] + RATES[(ORd_num_of_rates * offset) + nass] * dt;
  STATES[(ORd_num_of_states * offset) + ki] = STATES[(ORd_num_of_states * offset) + ki] + RATES[(ORd_num_of_rates * offset) + ki] * dt;
  STATES[(ORd_num_of_states * offset) + kss] = STATES[(ORd_num_of_states * offset) + kss] + RATES[(ORd_num_of_rates * offset) + kss] * dt;

  // if(offset == 1) printf("cai_rates (solve analytical): %lf\n",RATES[(ORd_num_of_rates * offset) + cai]);
  STATES[(ORd_num_of_states * offset) + cai] = STATES[(ORd_num_of_states * offset) + cai] + RATES[(ORd_num_of_rates * offset) + cai] * dt;
  // if(offset == 1) printf("cai_states (solve analytical): %lf\n",STATES[(ORd_num_of_states * offset) + cai]);

  STATES[(ORd_num_of_states * offset) + cass] = STATES[(ORd_num_of_states * offset) + cass] + RATES[(ORd_num_of_rates * offset) + cass] * dt;
  STATES[(ORd_num_of_states * offset) + cansr] = STATES[(ORd_num_of_states * offset) + cansr] + RATES[(ORd_num_of_rates * offset) + cansr] * dt;
  STATES[(ORd_num_of_states * offset) + cajsr] = STATES[(ORd_num_of_states * offset) + cajsr] + RATES[(ORd_num_of_rates * offset) + cajsr] * dt;
//for(int i=0;i<ORd_num_of_states;i++){
//    STATES[i] = STATES[i] + RATES[i] * dt;
//}
}

__device__ void ___gaussElimination(double *A, double *b, double *x, int N) {
        // Using A as a flat array to represent an N x N matrix
    for (int i = 0; i < N; i++) {
        // Search for maximum in this column
        double maxEl = fabs(A[i*N + i]);
        int maxRow = i;
        for (int k = i + 1; k < N; k++) {
            if (fabs(A[k*N + i]) > maxEl) {
                maxEl = fabs(A[k*N + i]);
                maxRow = k;
            }
        }

        // Swap maximum row with current row (column by column)
        for (int k = i; k < N; k++) {
            double tmp = A[maxRow*N + k];
            A[maxRow*N + k] = A[i*N + k];
            A[i*N + k] = tmp;
        }
        double tmp = b[maxRow];
        b[maxRow] = b[i];
        b[i] = tmp;

        // Make all rows below this one 0 in current column
        for (int k = i + 1; k < N; k++) {
            double c = -A[k*N + i] / A[i*N + i];
            for (int j = i; j < N; j++) {
                if (i == j) {
                    A[k*N + j] = 0;
                } else {
                    A[k*N + j] += c * A[i*N + j];
                }
            }
            b[k] += c * b[i];
        }
    }

    // Solve equation Ax=b for an upper triangular matrix A
    for (int i = N - 1; i >= 0; i--) {
        x[i] = b[i] / A[i*N + i];
        for (int k = i - 1; k >= 0; k--) {
            b[k] -= A[k*N + i] * x[i];
        }
    }
}

// void ohara_rudy_cipa_v1_2017::solveRK4(double TIME, double dt)
// {
// 	unsigned short idx;
// 	double k1[49],k2[49],k3[49],k4[49];
// 	double states_temp[49];
	
// 	computeRates(TIME, CONSTANTS, RATES, STATES, ALGEBRAIC );
// 	for(idx = 0; idx < ORd_num_of_states; idx++){
// 		k1[idx] = dt * RATES[idx];
// 		states_temp[idx] = STATES[idx] + k1[idx]*0.5;
// 	}
// 	computeRates(TIME+(dt*0.5), CONSTANTS, RATES, states_temp, ALGEBRAIC );
// 	for(idx = 0; idx < ORd_num_of_states; idx++){
// 		k2[idx] = dt * RATES[idx];
// 		states_temp[idx] = STATES[idx] + k2[idx]*0.5;
// 	}
// 	computeRates(TIME+(dt*0.5), CONSTANTS, RATES, states_temp, ALGEBRAIC );
// 	for(idx = 0; idx < ORd_num_of_states; idx++){
// 		k3[idx] = dt * RATES[idx];
// 		states_temp[idx] = STATES[idx] + k3[idx];
// 	}
// 	computeRates(TIME+dt, CONSTANTS, RATES, states_temp, ALGEBRAIC );
// 	for(idx = 0; idx < ORd_num_of_states; idx++){
// 		k4[idx] = dt * RATES[idx];
// 		STATES[idx] += (k1[idx]/6) + (k2[idx]/3) + (k3[idx]/3) + (k4[idx]/6) ;
// 	}
// }

__device__ double set_time_step (double TIME, double time_point, double max_time_step, double *CONSTANTS, double *RATES, int offset) {
 double min_time_step = 0.005;
 double time_step = min_time_step;
 double min_dV = 0.2;
 double max_dV = 0.8;
 
 if (TIME <= time_point || (TIME - floor(TIME / CONSTANTS[(ORd_num_of_constants * offset) +  BCL]) * CONSTANTS[(ORd_num_of_constants * offset) +  BCL]) <= time_point) {
    //printf("TIME <= time_point ms\n");
    return time_step;
    //printf("TIME = %E, dV = %E, time_step = %E\n",TIME, RATES[V] * time_step, time_step);
  }
  else {
    //printf("TIME > time_point ms\n");
    if (std::abs(RATES[(ORd_num_of_rates * offset) + V] * time_step) <= min_dV) {//Slow changes in V
        //printf("dV/dt <= 0.2\n");
        time_step = std::abs(max_dV / RATES[(ORd_num_of_rates * offset) + V]);
        //Make sure time_step is between min time step and max_time_step
        if (time_step < min_time_step) {
            time_step = min_time_step;
        }
        else if (time_step > max_time_step) {
            time_step = max_time_step;
        }
        //printf("TIME = %E, dV = %E, time_step = %E\n",TIME, RATES[V] * time_step, time_step);
    }
    else if (std::abs(RATES[(ORd_num_of_rates * offset) + V] * time_step) >= max_dV) {//Fast changes in V
        //printf("dV/dt >= 0.8\n");
        time_step = std::abs(min_dV / RATES[(ORd_num_of_rates * offset) + V]);
        //Make sure time_step is not less than 0.005
        if (time_step < min_time_step) {
            time_step = min_time_step;
        }
        //printf("TIME = %E, dV = %E, time_step = %E\n",TIME, RATES[V] * time_step, time_step);
    } else {
        time_step = min_time_step;
    }
    return time_step;
  }
}

/// using ord 2011 set time step
// __device__ double set_time_step(double TIME, double time_point, double max_time_step, double *CONSTANTS, double *RATES, int offset) {
//   double time_step = 0.005;
//   int num_of_constants = 146;
//   int num_of_rates = 41;

//   if (TIME <= time_point || (TIME - floor(TIME / CONSTANTS[BCL + (offset * num_of_constants)]) * CONSTANTS[BCL + (offset * num_of_constants)]) <= time_point) {
//     //printf("TIME <= time_point ms\n");
//     return time_step;
//     //printf("dV = %lf, time_step = %lf\n",RATES[V] * time_step, time_step);
//   }
//   else {
//     //printf("TIME > time_point ms\n");
//     if (std::abs(RATES[V + (offset * num_of_rates)] * time_step) <= 0.2) {//Slow changes in V
//         // printf("dV/dt <= 0.2\n");
//         time_step = std::abs(0.8 / RATES[V + (offset * num_of_rates)]);
//         //Make sure time_step is between 0.005 and max_time_step
//         if (time_step < 0.005) {
//             time_step = 0.005;
//         }
//         else if (time_step > max_time_step) {
//             time_step = max_time_step;
//         }
//         //printf("dV = %lf, time_step = %lf\n",std::abs(RATES[V] * time_step), time_step);
//     }
//     else if (std::abs(RATES[V + (offset * num_of_rates)] * time_step) >= 0.8) {//Fast changes in V
//         // printf("dV/dt >= 0.8\n");
//         time_step = std::abs(0.2 / RATES[V + (offset * num_of_rates)]);
//         while (std::abs(RATES[V + (offset * num_of_rates)]  * time_step) >= 0.8 &&
//                0.005 < time_step &&
//                time_step < max_time_step) {
//             time_step = time_step / 10.0;
//             // printf("dV = %lf, time_step = %lf\n",std::abs(RATES[V] * time_step), time_step);
//         }
//     }
//     // __syncthreads();
//     return time_step;
//   }
// }

