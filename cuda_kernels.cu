#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include "cuda_kernels.h"
#include "mc_growth.h"
#include <chrono>

#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}
axyz_work_item_t* dev_work_items;
axyz_result_t* dev_results;
float *dev_AA_ = nullptr, *dev_BB = nullptr, *dev_transform_array = nullptr;
curandState* dev_states = nullptr;
int *dev_ochered_count = nullptr, *dev_ochered_x = nullptr, *dev_ochered_y = nullptr, *dev_ochered_z = nullptr;

static int max_atoms_to_update_size = 0;
static int max_ochered_size_allocated = 0;
static double total_cuda_time_ms = 0.0;
static int g_optimal_block_size = 0;


__device__ void calc_neighbor_coords(int x, int y, int z, int dir, int Lx, int Ly, int Lz, int* x2, int* y2, int* z2) {
    int factor = (z % 2 == 0) ? 1 : -1;
    const int dx[16] = {1, 1, -1, -1, 0, 2, 2, 0, 2, 2, 0, -2, -2, 0, -2, -2};
    const int dy[16] = {1, -1, 1, -1, 2, 0, 2, -2, 0, -2, 2, 0, 2, -2, 0, -2};
    const int dz[16] = {1, -1, -1, 1, 2, 2, 0, -2, -2, 0, -2, -2, 0, 2, 2, 0};
    *x2 = (x + factor * dx[dir] + Lx) % Lx;
    *y2 = (y + factor * dy[dir] + Ly) % Ly;
    
    int new_z = z + factor * dz[dir];
    
    if (new_z < 2) new_z = 2;
    if (new_z >= Lz - 2) new_z = Lz - 3;
    *z2 = new_z;
}
__device__ void randn2_gpu(float* x1, float* x2, curandState* state) {
    float V1, V2, S, f;
    while (1) {
        V1 = 2.0f * curand_uniform(state) - 1.0f;
        V2 = 2.0f * curand_uniform(state) - 1.0f;
        S = V1 * V1 + V2 * V2;
        if (S < 1.0f && S > 0.0f) break;
    }
    f = sqrtf(-2.0f * logf(S) / S);
    *x1 = V1 * f;
    *x2 = V2 * f;
}

__device__ void random_displacements_gpu(float* ax_, float* ay_, float* az_, unsigned short int config_, float* transform_array_ptr, curandState* state, float T) {
    float a1, a2, a3, a4, coeff;
    coeff = sqrtf(0.5f * T);
    randn2_gpu(&a1, &a2, state);
    randn2_gpu(&a3, &a4, state);
    
    *ax_ = coeff * (transform_array_ptr[0] * a1 + transform_array_ptr[3] * a2 + transform_array_ptr[4] * a3);
    *ay_ = coeff * (transform_array_ptr[3] * a1 + transform_array_ptr[1] * a2 + transform_array_ptr[5] * a3);
    *az_ = coeff * (transform_array_ptr[4] * a1 + transform_array_ptr[5] * a2 + transform_array_ptr[2] * a3);
}

__global__ void setup_kernel(curandState* state, unsigned long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {return;}
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void axyz_kernel_packed(
    const axyz_work_item_t* work_items,
    axyz_result_t* results,
    int count,
    curandState* states,
    float T,
    float* d_AA_, float* d_BB, float* d_transform_array,
    int* d_ochered_x, int* d_ochered_y, int* d_ochered_z, int* d_ochered_count, int max_ochered_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    axyz_work_item_t item = work_items[idx];

    if (item.center.type == 0) {
        results[idx].a = {0.0f, 0.0f, 0.0f};
        return;
    }
    
    unsigned short config_ = item.center.config;
    
    float* AA_p = &d_AA_[config_ * 6];
    
    float A_xx = AA_p[0];
    float A_yy = AA_p[1];
    float A_zz = AA_p[2];
    float A_xy = AA_p[3];
    float A_xz = AA_p[4];
    float A_yz = AA_p[5];

    float Bx = item.center.B0.x;
    float By = item.center.B0.y;
    float Bz = item.center.B0.z;

    for (int dir = 0; dir < dir_number; dir++) {
        atom_t neighbor = item.neighbors[dir];
        
        float ax2 = neighbor.a.x;
        float ay2 = neighbor.a.y;
        float az2 = neighbor.a.z;

        float* BB_p = &d_BB[(config_ * dir_number + dir) * 9];

        Bx += BB_p[0] * ax2 + BB_p[1] * ay2 + BB_p[2] * az2;
        By += BB_p[3] * ax2 + BB_p[4] * ay2 + BB_p[5] * az2;
        Bz += BB_p[6] * ax2 + BB_p[7] * ay2 + BB_p[8] * az2;

        int ochered_idx = atomicAdd(d_ochered_count, 1);
        if (ochered_idx < max_ochered_size) {
            coord nb_coords = item.neighbor_coords[dir];
            d_ochered_x[ochered_idx] = nb_coords.x;
            d_ochered_y[ochered_idx] = nb_coords.y;
            d_ochered_z[ochered_idx] = nb_coords.z;        
        }
    }

    float ax_, ay_, az_;
    float* transform_p = &d_transform_array[config_ * 6];
    random_displacements_gpu(&ax_, &ay_, &az_, config_, transform_p, &states[idx], T);
        
    ax_ -= 0.5f * (A_xx * Bx + A_xy * By + A_xz * Bz);
    ay_ -= 0.5f * (A_xy * Bx + A_yy * By + A_yz * Bz);
    az_ -= 0.5f * (A_xz * Bx + A_yz * By + A_zz * Bz);

    results[idx].a.x = ax_;
    results[idx].a.y = ay_;
    results[idx].a.z = az_;

    int ochered_idx = atomicAdd(d_ochered_count, 1);
    if (ochered_idx < max_ochered_size) {
        d_ochered_x[ochered_idx] = item.center_coords.x;
        d_ochered_y[ochered_idx] = item.center_coords.y;
        d_ochered_z[ochered_idx] = item.center_coords.z;
    }
}

extern "C" void cuda_init(float* host_AA_, float* host_BB, float* host_transform_array, 
                          int max_atoms,float moves_percent) 
{
    // --- Global, static-sized arrays ---
    size_t atoms_size = Lx * Ly * Lz * sizeof(atom_t);
    size_t AA_size = Nconfig * 6 * sizeof(float);
    size_t BB_size = Nconfig * dir_number * 9 * sizeof(float);
    size_t transform_size = Nconfig * 6 * sizeof(float);
        
    cudaMalloc(&dev_AA_, AA_size);
    cudaMalloc(&dev_BB, BB_size);
    cudaMalloc(&dev_transform_array, transform_size);
    
    // Copy AA_, BB and transform_array data
    cudaMemcpy(dev_AA_, host_AA_, AA_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_BB, host_BB, BB_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_transform_array, host_transform_array, transform_size, cudaMemcpyHostToDevice);
    
    // --- Workspace buffers, sized for 2% of total atoms as a safety margin ---
    int atoms_to_update_size = (int)(max_atoms * (moves_percent/ 100.0f + 0.02f));
    if (atoms_to_update_size < 128) atoms_to_update_size = 128; // Set a minimum 
        
    cudaMalloc(&dev_work_items, atoms_to_update_size * sizeof(axyz_work_item_t));
    cudaMalloc(&dev_results, atoms_to_update_size * sizeof(axyz_result_t));

    printf("Allocating CUDA work buffers for up to %d atoms (2%% of total).\n", atoms_to_update_size);
    max_atoms_to_update_size = atoms_to_update_size;

    cudaMalloc(&dev_states, atoms_to_update_size * sizeof(curandState));
    max_ochered_size_allocated = atoms_to_update_size * (dir_number + 1);
    cudaMalloc(&dev_ochered_count, sizeof(int));
    cudaMalloc(&dev_ochered_x, max_ochered_size_allocated * sizeof(int));
    cudaMalloc(&dev_ochered_y, max_ochered_size_allocated * sizeof(int));
    cudaMalloc(&dev_ochered_z, max_ochered_size_allocated * sizeof(int));

    // Initialize random states for the entire buffer
    dim3 block_setup(128);
    dim3 grid_setup((atoms_to_update_size + block_setup.x - 1) / block_setup.x);
    setup_kernel<<<grid_setup, block_setup>>>(dev_states, 19, atoms_to_update_size);
    
    if (g_optimal_block_size == 0) {
        printf("Performing one-time CUDA occupancy calculation for 'axyz_kernel_packed'...\n");
        int max_occupancy = 0;
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        for (int block_size = 64; block_size <= prop.maxThreadsPerBlock; block_size += 32) {
            int active_blocks;
            
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_blocks,
                axyz_kernel_packed,
                block_size,
                0 // Static shared memory is known to the driver
            );
            
            int occupancy = active_blocks * block_size;
            if (occupancy > max_occupancy) {
                max_occupancy = occupancy;
                g_optimal_block_size = block_size;
            }
        }
        printf("CUDA Occupancy Calculation: Optimal block size set to %d\n", g_optimal_block_size);
    }
}


extern "C" void cuda_cleanup() {
    printf("\n===========================================================\n");
    printf("Total time spent in all cuda_do_many_axyz calls: %.4f ms\n", total_cuda_time_ms);
    printf("===========================================================\n\n");
    cudaFree(dev_AA_);
    cudaFree(dev_BB);
    cudaFree(dev_transform_array);
    cudaFree(dev_states);
    cudaFree(dev_ochered_count);
    cudaFree(dev_ochered_x);
    cudaFree(dev_ochered_y);
    cudaFree(dev_ochered_z);
    cudaFree(dev_work_items);
    cudaFree(dev_results);
}

extern "C" void cuda_do_many_axyz_packed(
    const axyz_work_item_t* host_work_items,
    axyz_result_t* host_results,
    int count,
    float T,
    int* host_ochered_x,
    int* host_ochered_y,
    int* host_ochered_z,
    int* out_host_ochered_count,
    int max_ochered_size
) {
    if (count == 0) {
        *out_host_ochered_count = 0;
        return;
    }
    if (count > max_atoms_to_update_size) {
        printf("\nFATAL CUDA ERROR: Trying to process %d atoms, but buffer is for %d\n", count, max_atoms_to_update_size);
        exit(EXIT_FAILURE);
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    cudaMemcpy(dev_work_items, host_work_items, count * sizeof(axyz_work_item_t), cudaMemcpyHostToDevice);

    int zero = 0;
    cudaMemcpy(dev_ochered_count, &zero, sizeof(int), cudaMemcpyHostToDevice);

    int block_size = g_optimal_block_size;
    dim3 block(block_size);
    dim3 grid((count + block.x - 1) / block.x);
    
    axyz_kernel_packed<<<grid, block>>>(
        dev_work_items, dev_results, count, dev_states, T,
        dev_AA_, dev_BB, dev_transform_array,
        dev_ochered_x, dev_ochered_y, dev_ochered_z, dev_ochered_count, max_ochered_size
    );
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    cudaMemcpy(host_results, dev_results, count * sizeof(axyz_result_t), cudaMemcpyDeviceToHost);
    
    int ochered_count_gpu;
    cudaMemcpy(&ochered_count_gpu, dev_ochered_count, sizeof(int), cudaMemcpyDeviceToHost);
    ochered_count_gpu = min(ochered_count_gpu, max_ochered_size);
    *out_host_ochered_count = ochered_count_gpu;
    if (ochered_count_gpu > 0) {
        cudaMemcpy(host_ochered_x, dev_ochered_x, ochered_count_gpu * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_ochered_y, dev_ochered_y, ochered_count_gpu * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_ochered_z, dev_ochered_z, ochered_count_gpu * sizeof(int), cudaMemcpyDeviceToHost);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_ms = end_time - start_time;
    total_cuda_time_ms += elapsed_ms.count();
}