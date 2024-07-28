#include "file_operations.hpp"
#include "../modules/glob_type.hpp"
#include "../modules/param.hpp"
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <regex>
#include <sys/stat.h>

/**
 * @brief Reads IC50 data and concentration values from a CSV file.
 *
 * @param file_name Name of the file containing IC50 data.
 * @param ic50 Array to store IC50 values.
 * @param conc Array to store concentration values.
 * @return int Number of samples read from the file.
 */
int get_IC50_data_from_file(const char *file_name, double *ic50, double *conc) {
    FILE *fp_drugs;
    char *token;
    char buffer_ic50[255];
    unsigned int idx_ic50 = 0, idx_conc = 0;
    int sample_size = 0;

    if ((fp_drugs = fopen(file_name, "r")) == NULL) {
        printf("Cannot open file %s\n", file_name);
        return 0;
    }

    // Skip header
    fgets(buffer_ic50, sizeof(buffer_ic50), fp_drugs);
    while (fgets(buffer_ic50, sizeof(buffer_ic50), fp_drugs) != NULL) {
        token = strtok(buffer_ic50, ",");
        conc[idx_conc++] = strtod(strtok(NULL, ","), NULL);
        token = strtok(NULL, ",");
        while (token != NULL) {
            ic50[idx_ic50++] = strtod(token, NULL);
            // printf("%s\n", token); // testingAuto
            token = strtok(NULL, ",");
        }
        sample_size++;
    }

    fclose(fp_drugs);
    return sample_size;
}

/**
 * @brief Reads conductance variability data from a CSV file.
 *
 * @param file_name Name of the file containing conductance variability data.
 * @param limit Maximum number of samples to read.
 * @param cvar Array to store conductance variability data.
 * @return int Number of samples read from the file.
 */
int get_cvar_data_from_file(const char *file_name, unsigned int limit, double *cvar) {
    char buffer_cvar[255];
    FILE *fp_cvar;
    char *token;
    unsigned int idx = 0;
    int sample_size = 0;

    if ((fp_cvar = fopen(file_name, "r")) == NULL) {
        printf("Cannot open file %s\n", file_name);
        return 0;
    }

    // Skip header
    fgets(buffer_cvar, sizeof(buffer_cvar), fp_cvar);
    while ((fgets(buffer_cvar, sizeof(buffer_cvar), fp_cvar) != NULL) && (sample_size < limit)) {
        token = strtok(buffer_cvar, ",");
        while (token != NULL) {
            cvar[idx++] = strtod(token, NULL);
            token = strtok(NULL, ",");
        }
        sample_size++;
    }

    fclose(fp_cvar);
    return sample_size;
}

/**
 * @brief Reads initial state data from a CSV file (acquiring init_state) and populates the provided array.
 *
 * This function opens a CSV file, reads the initial state data, and populates
 * the provided array with these values. It skips the header line and processes
 * each line of the file by tokenizing the comma-separated values.
 *
 * @param file_name The name of the file containing the initial state data.
 * @param init_states A pointer to an array where the initial state data will be stored.
 * @return int The number of samples read from the file.
 */

int get_init_data_from_file(const char *file_name, double *init_states) {
    // buffer for writing in snprintf() function
    char buffer_cache[1023];
    FILE *fp_cache;
    // cvar_t cvar;
    char *token;
    // std::array<double,18> temp_array;
    unsigned long idx;

    if ((fp_cache = fopen(file_name, "r")) == NULL) {
        printf("Cannot open file %s\n",
               file_name);
    }
    idx = 0;
    unsigned int sample_size = 0;
    fgets(buffer_cache, sizeof(buffer_cache), fp_cache); // skip header
    while ((fgets(buffer_cache, sizeof(buffer_cache), fp_cache) != NULL)) { // begin line reading
        token = strtok(buffer_cache, ",");
        while (token != NULL) { // begin data tokenizing
            init_states[idx++] = strtod(token, NULL);
            // if(idx < 82){
            //     printf("%d: %lf\n",idx-1,init_states[idx-1]);
            // }
            token = strtok(NULL, ",");
        } // end data tokenizing
        // printf("\n");
        sample_size++;
        // cvar.push_back(temp_array);
    } // end line reading

    fclose(fp_cache);
    return sample_size;
}

/**
 * @brief Extracts the drug name from the given filename.
 *
 * @param filename Name of the file.
 * @return char* Extracted drug name.
 */
char *get_drug_name(const char filename[1024]) {
    std::string path(filename);
    std::smatch match;
    if (std::regex_search(path, match, std::regex(R"((IC50_)?([a-zA-Z0-9_]+)\.csv$)"))) {
        std::string extracted_name = match[2].str();
        char *result = new char[extracted_name.size() + 1];
        std::strcpy(result, extracted_name.c_str());
        return result;
    } else {
        return nullptr;
    }
}

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
void write_results_to_file(const char *base_dir, const char *drug_name, double *h_states, cipa_t *h_cipa_result, int sample_size, int ORd_num_of_states) {
    printf("writing to file... \n");

    std::string base_path = std::string(base_dir) + "/init_" + drug_name + "/";

    if (mkdir(base_path.c_str(), 0777) == 0) {
        printf("Directory created\n");
    } else {
        printf("Unable to create directory\n");
    }

    std::string state_file = base_path + "state_only.csv";

    FILE *writer = fopen(state_file.c_str(), "w");
    if (writer == nullptr) {
        printf("Unable to open file for writing: %s\n", state_file.c_str());
        return;
    }
    fprintf(writer, "V,CaMKt,cass,nai,nass,ki,kss,cansr,cajsr,cai,m,hf,hs,j,hsp,jp,mL,hL,hLp,a,iF,iS,ap,iFp,iSp,d,ff,fs,fcaf,fcas,jca,ffp,fcafp,nca,xrf,xrs,xs1,xs2,xk1,Jrelnp,Jrelp,\n");
    for (int sample_id = 0; sample_id < sample_size; sample_id++) {
        for (int datapoint = 0; datapoint < ORd_num_of_states - 1; datapoint++) {
            fprintf(writer, "%lf,", h_states[(sample_id * ORd_num_of_states) + datapoint]);
        }
        fprintf(writer, "%lf\n", h_states[(sample_id * ORd_num_of_states) + ORd_num_of_states - 1]);
    }
    fclose(writer);

    std::string dvmdt_file = base_path + "dvmdt.csv";
    writer = fopen(dvmdt_file.c_str(), "w");
    if (writer == nullptr) {
        printf("Unable to open file for writing: %s\n", dvmdt_file.c_str());
        return;
    }
    fprintf(writer, "Sample,dVm/dt\n");
    for (int sample_id = 0; sample_id < sample_size; sample_id++) {
        fprintf(writer, "%d,%lf\n", sample_id, h_cipa_result[sample_id].dvmdt_repol);
    }
    fclose(writer);
}

int get_herg_data_from_file(const char* dir_name, char* drugname, double *herg)
{
  FILE *fp_herg;
  char *token;
  char full_herg_file_name[150];
  char buffer_herg[255];
  unsigned int idx;

  strcpy(full_herg_file_name, dir_name);
  strcat(full_herg_file_name,"/");
  strcat(drugname,".csv");
  strcat(full_herg_file_name,drugname);

  printf("reading herg file: %s\n",full_herg_file_name);

  if( (fp_herg = fopen(full_herg_file_name, "r")) == NULL){
    printf("Cannot open file %s\n", full_herg_file_name);
    return 0;
  }
  idx = 0;
  int sample_size = 0;
  fgets(buffer_herg, sizeof(buffer_herg), fp_herg); // skip header
  while( fgets(buffer_herg, sizeof(buffer_herg), fp_herg) != NULL )
    { // begin line reading
      token = strtok( buffer_herg, "," );
      while( token != NULL )
      { // begin data tokenizing
        herg[idx++] = strtod(token, NULL);
        token = strtok(NULL, ",");
      } // end data tokenizing
      sample_size++;
    } // end line reading

  fclose(fp_herg);
  printf("%lf, %lf, %lf, %lf, %lf, %lf\n",herg[0],herg[1],herg[2],herg[3],herg[4],herg[5]);
  return sample_size;
}