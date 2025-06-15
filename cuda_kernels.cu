#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include "cuda_kernels.h"
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
static int max_count = 0;
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
    float* p = transform_array + config_ * 6;
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

__global__ void setup_kernel(curandState* state, unsigned long seed, int max_n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_n) {return;}
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

    dev_atoms[atom_idx].config = massiv_to_config1(nb_type);
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

    if (atoms_read[atom_idx].type == 0) return;

    unsigned short config_ = atoms_write[atom_idx].config;
    if (config_ >= Nconfig || config_ == 65535) return;
    // ВОТ ТУТ НИЖЕ СКОРЕЕ ВСЕГО ЛАЖА
    __shared__ float shared_AA[6];
    __shared__ float shared_transform[6];
    if (threadIdx.x < 6) {
        shared_AA[threadIdx.x] = d_AA_[config_ * 6 + threadIdx.x];
        shared_transform[threadIdx.x] = d_transform_array[config_ * 6 + threadIdx.x];
    }
    __syncthreads();

    float A_xx = shared_AA[0];
    float A_yy = shared_AA[1];
    float A_zz = shared_AA[2];
    float A_xy = shared_AA[3];
    float A_xz = shared_AA[4];
    float A_yz = shared_AA[5];

    float Bx = atoms_read[atom_idx].B0.x;
    float By = atoms_read[atom_idx].B0.y;
    float Bz = atoms_read[atom_idx].B0.z;

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

            float* BB_ptr = d_BB + config_ * dir_number * 9 + dir * 9;
            Bx += BB_ptr[0] * ax2 + BB_ptr[1] * ay2 + BB_ptr[2] * az2;
            By += BB_ptr[3] * ax2 + BB_ptr[4] * ay2 + BB_ptr[5] * az2;
            Bz += BB_ptr[6] * ax2 + BB_ptr[7] * ay2 + BB_ptr[8] * az2;

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
    //И ТУТ ВОЗМОЖНО ЛАЖА и изза этого или сверху, выходим за границы!!!
    random_displacements_gpu(&ax_, &ay_, &az_, config_, shared_transform, &states[idx], T);

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
    } __syncthreads();
}

extern "C" void cuda_init(int Lx, int Ly, int Lz, atom_t* host_atoms, 
                          float* host_AA_, float* host_BB, float* host_transform_array, 
                          int max_atoms) 
{
    size_t atoms_size = Lx * Ly * Lz * sizeof(atom_t);
    size_t AA_size = Nconfig * 6 * sizeof(float);
    size_t BB_size = Nconfig * dir_number * 9 * sizeof(float);
    size_t transform_size = Nconfig * 6 * sizeof(float);
    max_ochered_size_allocated = max_atoms * (dir_number + 1);

    cudaMalloc(&dev_atoms_read, atoms_size);
    cudaMalloc(&dev_atoms_write, atoms_size);
    cudaMalloc(&dev_AA_, AA_size);
    cudaMalloc(&dev_BB, BB_size);
    cudaMalloc(&dev_transform_array, transform_size);
    cudaMalloc(&dev_states, max_atoms * sizeof(curandState));

    cudaMalloc(&dev_xs, max_atoms * sizeof(int));
    cudaMalloc(&dev_ys, max_atoms * sizeof(int));
    cudaMalloc(&dev_zs, max_atoms * sizeof(int));
    cudaMalloc(&dev_ochered_count, sizeof(int));
    cudaMalloc(&dev_ochered_x, max_ochered_size_allocated * sizeof(int));
    cudaMalloc(&dev_ochered_y, max_ochered_size_allocated * sizeof(int));
    cudaMalloc(&dev_ochered_z, max_ochered_size_allocated * sizeof(int));
    
    cudaMemcpy(dev_atoms_read, host_atoms, atoms_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_atoms_write, host_atoms, atoms_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_AA_, host_AA_, AA_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_BB, host_BB, BB_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_transform_array, host_transform_array, transform_size, cudaMemcpyHostToDevice);
    dim3 block_setup(128);
    dim3 grid_setup((max_atoms + block_setup.x - 1) / block_setup.x);
    setup_kernel<<<grid_setup, block_setup>>>(dev_states, 19, max_atoms);
    
    if (g_optimal_block_size == 0) {
        //printf("Performing one-time CUDA occupancy calculation...\n");
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
    // if (!atoms_to_update || !host_atoms || !host_AA_ || !host_BB || !host_transform_array) {
    //     printf("Ошибка: NULL указатель в cuda_do_many_axyz\n");
    //     return;
    // }
    // if (count <= 0 || count > max_count) {
    //     printf("cuda_do_many_axyz: invalid input: count=%d, max_count=%d\n", count, max_count);
    //     return;
    // }
    //printf("cuda_do_many_axyz: count=%d, Lx=%d, Ly=%d, Lz=%d\n", count, Lx, Ly, Lz);

    for (int i = 0; i < count; i++) {
        int x = atoms_to_update[i].x;
        int y = atoms_to_update[i].y;
        int z = atoms_to_update[i].z;
        int idx = get_atom_idx_host(x, y, z, Lx, Ly, Lz);
        if (idx < 0 || idx >= Lx * Ly * Lz) {
            printf("некорректный индекс %d для атома %d\n", idx, i);
            return;
        }
    }
    auto start_time = std::chrono::high_resolution_clock::now();

    // // Создание событий для замера времени
    // cudaEvent_t start_event, stop_event;
    // cudaEventCreate(&start_event);
    // cudaEventCreate(&stop_event);
    // cudaEventRecord(start_event, 0);

    cuda_sync_atoms(host_atoms, Lx, Ly, Lz);

    // // ======================= НАЧАЛО ОТЛАДОЧНОГО КОДА =======================
    // if (count > 0) {
    //     printf("\n--- DEBUG: Проверка содержимого dev_atoms_read ---\n");

    //     // 1. Берем первый атом из списка на обновление для теста
    //     int test_x = atoms_to_update[0].x;
    //     int test_y = atoms_to_update[0].y;
    //     int test_z = atoms_to_update[0].z;
    //     int test_idx = get_atom_idx_host(test_x, test_y, test_z, Lx, Ly, Lz);

    //     // 2. Находим его первого соседа (dir=0)
    //     int test_n_x, test_n_y, test_n_z;
    //     { // Локальный scope, чтобы calc_neighbor_coords не ругался на __device__
    //         int factor = (test_z % 2 == 0) ? 1 : -1;
    //         const int dx[16] = {1, 1, -1, -1, 0, 2, 2, 0, 2, 2, 0, -2, -2, 0, -2, -2};
    //         const int dy[16] = {1, -1, 1, -1, 2, 0, 2, -2, 0, -2, 2, 0, 2, -2, 0, -2};
    //         const int dz[16] = {1, -1, -1, 1, 2, 2, 0, -2, -2, 0, -2, -2, 0, 2, 2, 0};
    //         test_n_x = (test_x + factor * dx[0] + Lx) % Lx;
    //         test_n_y = (test_y + factor * dy[0] + Ly) % Ly;
    //         int new_z = test_z + factor * dz[0];
    //         if (new_z < 2) new_z = 2;
    //         if (new_z >= Lz - 2) new_z = Lz - 3;
    //         test_n_z = new_z;
    //     }
    //     int test_neighbor_idx = get_atom_idx_host(test_n_x, test_n_y, test_n_z, Lx, Ly, Lz);

    //     // 3. Копируем данные этих двух атомов с GPU (dev_atoms_read)
    //     atom_t gpu_atom, gpu_neighbor;
    //     CUDA_CHECK(cudaMemcpy(&gpu_atom, &dev_atoms_read[test_idx], sizeof(atom_t), cudaMemcpyDeviceToHost));
    //     CUDA_CHECK(cudaMemcpy(&gpu_neighbor, &dev_atoms_read[test_neighbor_idx], sizeof(atom_t), cudaMemcpyDeviceToHost));
        
    //     // 4. Берем те же данные напрямую с CPU (host_atoms) для сравнения
    //     atom_t cpu_atom = host_atoms[test_idx];
    //     atom_t cpu_neighbor = host_atoms[test_neighbor_idx];

    //     // 5. Печатаем и сравниваем
    //     printf("Тестовый атом (idx: %d):\n", test_idx);
    //     printf("  CPU : type=%d, a.x=%.4f\n", cpu_atom.type, cpu_atom.a.x);
    //     printf("  GPU : type=%d, a.x=%.4f\n", gpu_atom.type, gpu_atom.a.x);
        
    //     printf("Его сосед (idx: %d):\n", test_neighbor_idx);
    //     printf("  CPU : type=%d, a.x=%.4f\n", cpu_neighbor.type, cpu_neighbor.a.x);
    //     printf("  GPU : type=%d, a.x=%.4f\n", gpu_neighbor.type, gpu_neighbor.a.x);

    //     if (cpu_atom.type == gpu_atom.type && cpu_neighbor.type == gpu_neighbor.type) {
    //         printf("РЕЗУЛЬТАТ: Успех! Данные на GPU полностью совпадают с CPU. Соседи на месте.\n");
    //     } else {
    //         printf("РЕЗУЛЬТАТ: Ошибка! Данные на GPU не совпадают. Проблема в копировании.\n");
    //     }
    //     printf("--------------------------------------------------\n\n");
    // }
    // // ======================== КОНЕЦ ОТЛАДОЧНОГО КОДА =======================

    int max_ochered_size = count * (dir_number + 1); // Каждый атом + до dir_number соседей
    if (max_ochered_size > max_ochered_size_allocated) {
        // We can reallocate here if needed, for now just print an error
        printf("Error: max_ochered_size (%d) exceeds allocated size (%d)\n", max_ochered_size, max_ochered_size_allocated);
        return;
    }

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
        // For a small number of atoms, round up to the nearest multiple of 32
        block_size = ((count + 31) / 32) * 32;
        // Ensure block_size is at least 32 if there are any atoms
        if (block_size == 0) block_size = 32;
        printf("block size = %d",block_size);
    } else {
        // For a large number of atoms, use the pre-calculated optimal block size
        block_size = g_optimal_block_size;
                printf("block size = %d",block_size);

    }

    // Ensure g_optimal_block_size has been initialized
    if (block_size <= 0) {
        block_size = 128; // Fallback to a default value
        printf("Warning: optimal block size not set or invalid. Falling back to %d.\n", block_size);
    }

    dim3 block(block_size);
    dim3 grid((count + block.x - 1) / block.x);

    //printf("Dynamic CUDA Launch: Atoms=%d -> BlockSize=%d, GridSize=%d\n", count, block_size, grid.x);


    // dim3 setup_grid((count + 127) / 128);
    // dim3 setup_block(128);
    // setup_kernel<<<setup_grid, setup_block>>>(dev_states, 19);
    int zero = 0;
    cudaMemcpy(dev_ochered_count, &zero, sizeof(int), cudaMemcpyHostToDevice);

    set_config_kernel<<<grid, block>>>(dev_atoms_write, Lx, Ly, Lz, dev_xs, dev_ys, dev_zs, count);
    //cudaDeviceSynchronize();

    axyz_kernel<<<grid, block>>>(dev_atoms_read, dev_atoms_write, Lx, Ly, Lz, dev_xs, dev_ys, dev_zs, count, dev_states, T,
                                 dev_AA_, dev_BB, dev_transform_array,
                                 dev_ochered_x, dev_ochered_y, dev_ochered_z, dev_ochered_count, max_ochered_size);
    
    // cudaEventRecord(stop_event, 0);
    // cudaEventSynchronize(stop_event);

    // float cuda_time_ms;
    // cudaEventElapsedTime(&cuda_time_ms, start_event, stop_event);
    // total_cuda_time_ms += cuda_time_ms;
    cudaDeviceSynchronize();

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