#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include "mc_growth.h"
#include <vector_types.h>

extern atom_t* dev_atoms_read;
extern atom_t* dev_atoms_write;
extern atom_t* dev_atoms;
extern float *dev_AA_, *dev_BB, *dev_transform_array;
extern curandState* dev_states;

//int get_atom_idx_host(int x, int y, int z, int Lx, int Ly, int Lz);

// Эта структура инкапсулирует все данные, необходимые для одного вычисления axyz.
// Она будет подготавливаться на CPU и отправляться на GPU.
struct axyz_work_item_t {
    atom_t center;
    atom_t neighbors[dir_number];
    coord  center_coords;
    coord  neighbor_coords[dir_number];
    // float random_numbers[4];
};

// Эта структура содержит результат одного вычисления axyz.
// Она будет отправлена обратно с GPU на CPU.
struct axyz_result_t {
    float3 a;
};

extern "C" {
    void cuda_init(float* host_AA, float* host_BB, float* host_transform_array, int max_total_atoms, float moves_percent);
    void cuda_cleanup();
    void cuda_sync_atoms(atom_t* host_atoms, int Lx, int Ly, int Lz);
    void cuda_do_many_axyz(struct coord* atoms_to_update, int count, atom_t* host_atoms, int Lx, int Ly, int Lz, float T,
                           float* host_AA_, float* host_BB, float* host_transform_array);
    void cuda_do_many_axyz_packed(
        const axyz_work_item_t* host_work_items,
        axyz_result_t* host_results,
        int count,
        float T,
        int* host_ochered_x,
        int* host_ochered_y,
        int* host_ochered_z,
        int* out_host_ochered_count,
        int max_ochered_size
    );
}

#endif