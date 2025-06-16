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

atom_t* dev_atoms_read = nullptr;
atom_t* dev_atoms_write = nullptr;
float *dev_AA_ = nullptr, *dev_BB = nullptr, *dev_transform_array = nullptr;
curandState* dev_states = nullptr;
int *dev_xs = nullptr, *dev_ys = nullptr, *dev_zs = nullptr;
int *dev_ochered_count = nullptr, *dev_ochered_x = nullptr, *dev_ochered_y = nullptr, *dev_ochered_z = nullptr;

static int max_atoms_to_update_size = 0;
static int max_ochered_size_allocated = 0;
static double total_cuda_time_ms = 0.0;
static int g_optimal_block_size = 0;


__host__ __device__ void convert(int x, int y, int z, int Lx, int Ly, int &lx, int &ly, int &lz) {
    lx = x;
    ly = ((y >> 2) << 1) + ((y & 0x3) >> 1);
    lz = z;
}
__host__ __device__ int get_atom_idx(int x, int y, int z, int Lx, int Ly, int Lz) {
    int lx, ly, lz;
    convert(x, y, z, Lx, Ly, lx, ly, lz);
    return (long)lz * (Ly / 2) * Lx + (long)ly * Lx + lx;
}
__host__ void convert_host(int x, int y, int z, int Lx, int Ly, int &lx, int &ly, int &lz) {
    lx = x;
    ly = ((y >> 2) << 1) + ((y & 0x3) >> 1);
    lz = z;
}

__host__ int get_atom_idx_host(int x, int y, int z, int Lx, int Ly, int Lz) {
    int lx, ly, lz;
    convert_host(x, y, z, Lx, Ly, lx, ly, lz);
    return (long)lz * (Ly / 2) * Lx + (long)ly * Lx + lx;
}

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
        if (S < 1.0f) break;
    }
    f = sqrtf(-2.0f * logf(S) / S);
    *x1 = V1 * f;
    *x2 = V2 * f;
}

__device__ void random_displacements_gpu(float* ax_, float* ay_, float* az_, unsigned short int config_, float* transform_array, curandState* state, float T) {
    if (config_ >= Nconfig || config_ == 65535 || T <= 0) {
        *ax_ = 0.0f; *ay_ = 0.0f; *az_ = 0.0f;
        return;
    }
    float a1, a2, a3, a4, coeff;
    coeff = sqrtf(0.5f * T);
    randn2_gpu(&a1, &a2, state);
    randn2_gpu(&a3, &a4, state);
    
    // Получаем указатель на начало массива для текущего config_, как в последовательной версии
    float* p = &transform_array[config_ * 6];
    
    // Используем тот же подход, что и в последовательной версии
    *ax_ = coeff * (p[0] * a1 + p[3] * a2 + p[4] * a3);
    *ay_ = coeff * (p[3] * a1 + p[1] * a2 + p[5] * a3);
    *az_ = coeff * (p[4] * a1 + p[5] * a2 + p[2] * a3);
}
__device__ unsigned short massiv_to_config1(int* nb_type) {
    unsigned short conf = 0;
    for (int i = 0; i < dir_number; i++) {
        conf <<= 1;
        conf += (nb_type[i] != 0);
    }
    return conf;
}

__global__ void setup_kernel(curandState* state, unsigned long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {return;}
    curand_init(seed, idx, 0, &state[idx]);
}


__global__ void set_config_kernel(
    atom_t* dev_atoms, int Lx, int Ly, int Lz,
    int* xs, int* ys, int* zs, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    int x = xs[idx];
    int y = ys[idx];
    int z = zs[idx];
    int atom_idx = get_atom_idx(x, y, z, Lx, Ly, Lz);

    int nb_type[16];
    for (int dir = 0; dir < dir_number; dir++) {
        int x2, y2, z2;
        calc_neighbor_coords(x, y, z, dir, Lx, Ly, Lz, &x2, &y2, &z2);
        int nb_idx = get_atom_idx(x2, y2, z2, Lx, Ly, Lz);
        nb_type[dir] = dev_atoms[nb_idx].type;
    }

    unsigned short new_config = massiv_to_config1(nb_type);
    dev_atoms[atom_idx].config = new_config;

    // // Debug print для первых 10 потоков
    // if (idx < 10) {
    //     printf("set_config_kernel - Thread %d: atom at (%d,%d,%d) got new config_=%d\n", 
    //            idx, x, y, z, new_config);
        
    //     // Выводим типы соседей
    //     printf("  Neighbor types: ");
    //     for (int dir = 0; dir < dir_number; dir++) {
    //         printf("%d ", nb_type[dir]);
    //     }
    //     printf("\n");
    // }
}

// Ядро для обновления смещений атомов
__global__ void axyz_kernel(
    atom_t* atoms_read, atom_t* atoms_write, int Lx, int Ly, int Lz,
    int* xs, int* ys, int* zs, int count, curandState* states, float T,
    float* d_AA_, float* d_BB, float* d_transform_array,
    int* d_ochered_x, int* d_ochered_y, int* d_ochered_z, int* d_ochered_count, int max_ochered_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    int x = xs[idx];
    int y = ys[idx];
    int z = zs[idx];
    int atom_idx = get_atom_idx(x, y, z, Lx, Ly, Lz);

    // Добавь проверку на выход за границы
   
    if (atoms_read[atom_idx].type == 0) return;

    unsigned short config_ = atoms_write[atom_idx].config;
    if (config_ >= Nconfig || config_ == 65535) {
        //printf("Ошибка: config_=%d превышает Nconfig=%d или равно 65535 (idx=%d)\n", config_, Nconfig, idx);
        return;
    }
    if (config_ * 6 + 5 >= Nconfig * 6) {
        //printf("Ошибка: config_=%d выходит за пределы d_AA_ или d_transform_array (Nconfig=%d, idx=%d)\n", config_, Nconfig, idx);
        return;
    }

    // // Debug print для всех потоков
    // if (idx < 10) { // Выводим для первых 10 потоков
    //     printf("Thread %d: atom at (%d,%d,%d) has config_=%d\n", 
    //            idx, x, y, z, config_);
    // }

    // Получаем указатели на начало массивов для текущего config_
    float* AA_p = &d_AA_[config_ * 6];
    float* transform_p = &d_transform_array[config_ * 6];

    // Прямой доступ к элементам массива, как в последовательной версии
    float A_xx = AA_p[0];
    float A_yy = AA_p[1];
    float A_zz = AA_p[2];
    float A_xy = AA_p[3];
    float A_xz = AA_p[4];
    float A_yz = AA_p[5];

    // // Debug print
    // if (idx == 0) {
    //     printf("Debug: config_=%d, AA_p[0]=%f,AA_p[1]=%f,AA_p[2]=%f,AA_p[3]=%f,AA_p[4]=%f,AA_p[5]=%f, transform_p[0]=%f\n", 
    //            config_, AA_p[0],AA_p[1],AA_p[2],AA_p[2],AA_p[3],AA_p[4],AA_p[5], transform_p[0]);
    // }

    float Bx = atoms_read[atom_idx].B0.x;
    float By = atoms_read[atom_idx].B0.y;
    float Bz = atoms_read[atom_idx].B0.z;

    // Debug print для начальных значений B
    // if (idx == 0) {
    //     printf("Debug: Initial B values: Bx=%f, By=%f, Bz=%f\n", Bx, By, Bz);
    // }

    for (int dir = 0; dir < dir_number; dir++) {
        int x2, y2, z2;
        calc_neighbor_coords(x, y, z, dir, Lx, Ly, Lz, &x2, &y2, &z2);
        int neighbor_idx = get_atom_idx(x2, y2, z2, Lx, Ly, Lz);

        if (atoms_read[neighbor_idx].type == 0) {
            continue;
        }
        
        if (neighbor_idx >= 0 && neighbor_idx < Lx * Ly * Lz) {
            float ax2 = atoms_read[neighbor_idx].a.x;
            float ay2 = atoms_read[neighbor_idx].a.y;
            float az2 = atoms_read[neighbor_idx].a.z;

            // Получаем указатель на нужный блок BB для текущего config_ и dir
            float* BB_p = &d_BB[(config_ * dir_number + dir) * 9];

            // Debug print для BB значений
            if (idx == 0 && dir == 0) {
                printf("Debug: config_=%d, dir=%d, BB values: %f,%f,%f,%f,%f,%f,%f,%f,%f\n", 
                       config_, dir, BB_p[0], BB_p[1], BB_p[2], BB_p[3], BB_p[4], 
                       BB_p[5], BB_p[6], BB_p[7], BB_p[8]);
            }

            Bx += BB_p[0] * ax2 + BB_p[1] * ay2 + BB_p[2] * az2;
            By += BB_p[3] * ax2 + BB_p[4] * ay2 + BB_p[5] * az2;
            Bz += BB_p[6] * ax2 + BB_p[7] * ay2 + BB_p[8] * az2;

            // Debug print для обновленных значений B после каждого соседа
            // if (idx == 0) {
            //     printf("Debug: After neighbor dir=%d: Bx=%f, By=%f, Bz=%f\n", dir, Bx, By, Bz);
            // }

            // соседа в очередь
            int ochered_idx = atomicAdd(d_ochered_count, 1);
            if (ochered_idx < max_ochered_size) {
                d_ochered_x[ochered_idx] = x2;
                d_ochered_y[ochered_idx] = y2;
                d_ochered_z[ochered_idx] = z2;
            }
        }
    }

    float ax_, ay_, az_;
    random_displacements_gpu(&ax_, &ay_, &az_, config_, transform_p, &states[idx], T);
    
    // Debug print
    // if (idx == 0) {
    //     printf("Debug: After random_displacements_gpu: ax_=%f, ay_=%f, az_=%f\n", ax_, ay_, az_);
    // }

    ax_ -= 0.5f * (A_xx * Bx + A_xy * By + A_xz * Bz);
    ay_ -= 0.5f * (A_xy * Bx + A_yy * By + A_yz * Bz);
    az_ -= 0.5f * (A_xz * Bx + A_yz * By + A_zz * Bz);

    atoms_write[atom_idx].a.x = ax_;
    atoms_write[atom_idx].a.y = ay_;
    atoms_write[atom_idx].a.z = az_;
    atoms_write[atom_idx].type = atoms_read[atom_idx].type;
    atoms_write[atom_idx].B0 = atoms_read[atom_idx].B0;

    // Добавляем текущий атом в очередь
    int ochered_idx = atomicAdd(d_ochered_count, 1);
    if (ochered_idx < max_ochered_size) {
        d_ochered_x[ochered_idx] = x;
        d_ochered_y[ochered_idx] = y;
        d_ochered_z[ochered_idx] = z;
    }
}

extern "C" void cuda_init(int Lx, int Ly, int Lz, atom_t* host_atoms, 
                          float* host_AA_, float* host_BB, float* host_transform_array, 
                          int max_atoms) 
{
    // --- Global, static-sized arrays ---
    size_t atoms_size = Lx * Ly * Lz * sizeof(atom_t);
    size_t AA_size = Nconfig * 6 * sizeof(float);
    size_t BB_size = Nconfig * dir_number * 9 * sizeof(float);
    size_t transform_size = Nconfig * 6 * sizeof(float);

    cudaMalloc(&dev_atoms_read, atoms_size);
    cudaMalloc(&dev_atoms_write, atoms_size);
    cudaMalloc(&dev_AA_, AA_size);
    cudaMalloc(&dev_BB, BB_size);
    cudaMalloc(&dev_transform_array, transform_size);
    
    cudaMemcpy(dev_atoms_read, host_atoms, atoms_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_atoms_write, host_atoms, atoms_size, cudaMemcpyHostToDevice);
    
    // Copy AA_, BB and transform_array data
    cudaMemcpy(dev_AA_, host_AA_, AA_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_BB, host_BB, BB_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_transform_array, host_transform_array, transform_size, cudaMemcpyHostToDevice);
    
    // --- Workspace buffers, sized for 2% of total atoms as a safety margin ---
    int atoms_to_update_size = (int)(max_atoms * 0.02);
    if (atoms_to_update_size < 128) atoms_to_update_size = 128; // Set a minimum 
    
    printf("Allocating CUDA work buffers for up to %d atoms (2%% of total).\n", atoms_to_update_size);
    max_atoms_to_update_size = atoms_to_update_size;

    cudaMalloc(&dev_states, atoms_to_update_size * sizeof(curandState));
    cudaMalloc(&dev_xs, atoms_to_update_size * sizeof(int));
    cudaMalloc(&dev_ys, atoms_to_update_size * sizeof(int));
    cudaMalloc(&dev_zs, atoms_to_update_size * sizeof(int));

    max_ochered_size_allocated = atoms_to_update_size * (dir_number + 1);
    cudaMalloc(&dev_ochered_count, sizeof(int));
    cudaMalloc(&dev_ochered_x, max_ochered_size_allocated * sizeof(int));
    cudaMalloc(&dev_ochered_y, max_ochered_size_allocated * sizeof(int));
    cudaMalloc(&dev_ochered_z, max_ochered_size_allocated * sizeof(int));

    // Initialize random states for the entire buffer
    dim3 block_setup(128);
    dim3 grid_setup((atoms_to_update_size + block_setup.x - 1) / block_setup.x);
    setup_kernel<<<grid_setup, block_setup>>>(dev_states, 19, atoms_to_update_size);
    
    // --- One-time occupancy calculation ---
    if (g_optimal_block_size == 0) {
        printf("Performing one-time CUDA occupancy calculation...\n");
        int max_occupancy = 0;
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        for (int block_size = 64; block_size <= prop.maxThreadsPerBlock; block_size += 32) {
            int active_blocks;
            
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_blocks,
                axyz_kernel,
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
    cudaFree(dev_atoms_read);
    cudaFree(dev_atoms_write);
    cudaFree(dev_AA_);
    cudaFree(dev_BB);
    cudaFree(dev_transform_array);
    cudaFree(dev_states);
    cudaFree(dev_xs);
    cudaFree(dev_ys);
    cudaFree(dev_zs);
    cudaFree(dev_ochered_count);
    cudaFree(dev_ochered_x);
    cudaFree(dev_ochered_y);
    cudaFree(dev_ochered_z);
}

extern "C" void cuda_sync_atoms(atom_t* host_atoms, int Lx, int Ly, int Lz) {
    size_t atoms_size = Lx * Ly * Lz * sizeof(atom_t);
    cudaMemcpy(dev_atoms_read, host_atoms, atoms_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_atoms_write, host_atoms, atoms_size, cudaMemcpyHostToDevice);
}

extern "C" void cuda_do_many_axyz(
    struct coord* atoms_to_update,
    int count,
    atom_t* host_atoms,
    int Lx, int Ly, int Lz,
    float T,
    float* host_AA_,
    float* host_BB,
    float* host_transform_array)
{
    if (count == 0) {
        return;
    }

    // // CRITICAL: Check if the number of atoms to process exceeds our pre-allocated buffer.
    // if (count > max_atoms_to_update_size) {
    //     printf("\nFATAL CUDA ERROR:\n");
    //     printf("  Attempting to process %d atoms, but the work buffer was only allocated for %d.\n", count, max_atoms_to_update_size);
    //     printf("  This usually means 'param.moves_percent' is larger than the 2%% buffer size.\n");
    //     printf("  Solution: Increase the percentage in cuda_init() or decrease 'param.moves_percent'.\n\n");
    //     exit(EXIT_FAILURE);
    // }
   
    auto start_time = std::chrono::high_resolution_clock::now();

    cuda_sync_atoms(host_atoms, Lx, Ly, Lz);
    
    // Sync AA_, BB and transform_array data
    size_t AA_size = Nconfig * 6 * sizeof(float);
    size_t BB_size = Nconfig * dir_number * 9 * sizeof(float);
    size_t transform_size = Nconfig * 6 * sizeof(float);
    cudaMemcpy(dev_AA_, host_AA_, AA_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_BB, host_BB, BB_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_transform_array, host_transform_array, transform_size, cudaMemcpyHostToDevice);

    int* host_xs = new int[count];
    int* host_ys = new int[count];
    int* host_zs = new int[count];
    for (int i = 0; i < count; i++) {
        host_xs[i] = atoms_to_update[i].x;
        host_ys[i] = atoms_to_update[i].y;
        host_zs[i] = atoms_to_update[i].z;
    }
    cudaMemcpy(dev_xs, host_xs, count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ys, host_ys, count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_zs, host_zs, count * sizeof(int), cudaMemcpyHostToDevice);
    delete[] host_xs;
    delete[] host_ys;
    delete[] host_zs;

    int block_size;
    if (count < g_optimal_block_size) {
        block_size = ((count + 31) / 32) * 32;
        if (block_size == 0) block_size = 32;
    } else {
        block_size = g_optimal_block_size;
    }

    // Ensure g_optimal_block_size has been initialized
    if (block_size <= 0) {
        block_size = 128; 
        printf("Warning: optimal block size not set or invalid. Falling back to %d.\n", block_size);
    }

    dim3 block(block_size);
    dim3 grid((count + block.x - 1) / block.x);

    printf("Dynamic CUDA Launch: Atoms=%d -> BlockSize=%d, GridSize=%d\n", count, block_size, grid.x);
    int max_ochered_size = count * (dir_number + 1);
    int zero = 0;
    cudaMemcpy(dev_ochered_count, &zero, sizeof(int), cudaMemcpyHostToDevice);

    axyz_kernel<<<grid, block>>>(dev_atoms_read, dev_atoms_write, Lx, Ly, Lz, dev_xs, dev_ys, dev_zs, count, dev_states, T,
                                 dev_AA_, dev_BB, dev_transform_array,
                                 dev_ochered_x, dev_ochered_y, dev_ochered_z, dev_ochered_count, max_ochered_size);
    CUDA_CHECK(cudaGetLastError());

    // cudaEventRecord(stop_event, 0);
    // cudaEventSynchronize(stop_event);

    // float cuda_time_ms;
    // cudaEventElapsedTime(&cuda_time_ms, start_event, stop_event);
    // total_cuda_time_ms += cuda_time_ms;
    CUDA_CHECK(cudaDeviceSynchronize());

    // Копируем обновлённые атомы обратно на хост
    for (int i = 0; i < count; i++) {
        int x = atoms_to_update[i].x;
        int y = atoms_to_update[i].y;
        int z = atoms_to_update[i].z;
        int idx = get_atom_idx_host(x, y, z, Lx, Ly, Lz);
        cudaMemcpy(&host_atoms[idx], &dev_atoms_write[idx], sizeof(atom_t), cudaMemcpyDeviceToHost);
    }

    int ochered_count;
    cudaMemcpy(&ochered_count, dev_ochered_count, sizeof(int), cudaMemcpyDeviceToHost);
    ochered_count = min(ochered_count, max_ochered_size);
    //printf("ochered_count=%d, max_ochered_size=%d\n", ochered_count, max_ochered_size);

    int* host_ochered_x = new int[max_ochered_size];
    int* host_ochered_y = new int[max_ochered_size];
    int* host_ochered_z = new int[max_ochered_size];
    cudaMemcpy(host_ochered_x, dev_ochered_x, max_ochered_size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_ochered_y, dev_ochered_y, max_ochered_size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_ochered_z, dev_ochered_z, max_ochered_size * sizeof(int), cudaMemcpyDeviceToHost);

    for (int j = 0; j < ochered_count; j++) {
        if (host_ochered_x[j] < 0 || host_ochered_x[j] >= Lx ||
            host_ochered_y[j] < 0 || host_ochered_y[j] >= Ly ||
            host_ochered_z[j] < 0 || host_ochered_z[j] >= Lz) {
            // printf(" координаты (%d, %d, %d) на j=%d\n",
            //        host_ochered_x[j], host_ochered_y[j], host_ochered_z[j], j);
            continue;
        }
        v_ochered_Edef(host_ochered_x[j], host_ochered_y[j], host_ochered_z[j]);
    }
    delete[] host_ochered_x;
    delete[] host_ochered_y;
    delete[] host_ochered_z;
    // cudaEventDestroy(start_event);
    // cudaEventDestroy(stop_event);

    //=================== КОНЕЦ ИЗМЕРЕНИЯ ВРЕМЕНИ ===================
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_ms = end_time - start_time;
    total_cuda_time_ms += elapsed_ms.count();
}