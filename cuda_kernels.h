#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include "mc_growth.h"

extern atom_t* dev_atoms;
extern float *dev_AA_, *dev_BB, *dev_transform_array;
extern curandState* dev_states;

extern "C" void cuda_init(int Lx, int Ly, int Lz, atom_t* host_atoms, float* host_AA_, float* host_BB, float* host_transform_array, int max_atoms);
extern "C" void cuda_cleanup();
extern "C" void cuda_sync_atoms(atom_t* host_atoms, int Lx, int Ly, int Lz);
extern "C" void cuda_do_many_axyz(struct coord* atoms_to_update, int count, atom_t* host_atoms, int Lx, int Ly, int Lz, float T, float* host_AA_, float* host_BB, float* host_transform_array);

#endif