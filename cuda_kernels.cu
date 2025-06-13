#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include "cuda_kernels.h"
#include <chrono>

atom_t* dev_atoms_read = nullptr;
atom_t* dev_atoms_write = nullptr;
float *dev_AA_ = nullptr, *dev_BB = nullptr, *dev_transform_array = nullptr;
curandState* dev_states = nullptr;
static int max_count = 0;
    auto start = std::chrono::high_resolution_clock::now();

__host__ __device__ void convert(int x, int y, int z, int Lx, int Ly, int &lx, int &ly, int &lz) {
    lx = x; ly = ((y >> 2) << 1) + ((y & 0x3) >> 1); lz = z >> 2;
}
__host__ __device__ int get_atom_idx(int x, int y, int z, int Lx, int Ly, int Lz) {
    int lx, ly, lz; convert(x, y, z, Lx, Ly, lx, ly, lz);
    return lz * (Lx * (Ly / 2)) + ly * Lx + lx;
}
__host__ void convert_host(int x, int y, int z, int Lx, int Ly, int &lx, int &ly, int &lz) {
    lx = x;
    ly = ((y >> 2) << 1) + ((y & 0x3) >> 1);
    lz = z >> 2;
}

__host__ int get_atom_idx_host(int x, int y, int z, int Lx, int Ly, int Lz) {
    int lx, ly, lz;
    convert_host(x, y, z, Lx, Ly, lx, ly, lz);
    return lz * (Lx * (Ly / 2)) + ly * Lx + lx;
}

__device__ void calc_neighbor_coords(int x, int y, int z, int dir, int Lx, int Ly, int Lz, int* x2, int* y2, int* z2) {
    int factor = (z % 2 == 0) ? 1 : -1;
    const int dx[16] = {1, 1, -1, -1, 0, 2, 2, 0, 2, 2, 0, -2, -2, 0, -2, -2};
    const int dy[16] = {1, -1, 1, -1, 2, 0, 2, -2, 0, -2, 2, 0, 2, -2, 0, -2};
    const int dz[16] = {1, -1, -1, 1, 2, 2, 0, -2, -2, 0, -2, -2, 0, 2, 2, 0};
    *x2 = (x + factor * dx[dir] + Lx) % Lx;
    *y2 = (y + factor * dy[dir] + Ly) % Ly;
    *z2 = max(0, min(Lz - 1, z + factor * dz[dir]));
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

__global__ void setup_kernel(curandState* state, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
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
    random_displacements_gpu(&ax_, &ay_, &az_, config_, shared_transform, &states[idx], T);

    ax_ -= 0.5f * (A_xx * Bx + A_xy * By + A_xz * Bz);
    ay_ -= 0.5f * (A_xy * Bx + A_yy * By + A_yz * Bz);
    az_ -= 0.5f * (A_xz * Bx + A_yz * By + A_zz * Bz);

    atoms_write[atom_idx].a.x = ax_;
    atoms_write[atom_idx].a.y = ay_;
    atoms_write[atom_idx].a.z = az_;
    atoms_write[atom_idx].type = atoms_read[atom_idx].type;
    atoms_write[atom_idx].config = atoms_read[atom_idx].config;
    atoms_write[atom_idx].B0 = atoms_read[atom_idx].B0;

    // Добавляем текущий атом в очередь
    int ochered_idx = atomicAdd(d_ochered_count, 1);
    if (ochered_idx < max_ochered_size) {
        d_ochered_x[ochered_idx] = x;
        d_ochered_y[ochered_idx] = y;
        d_ochered_z[ochered_idx] = z;
    }
}

extern "C" void cuda_init(int Lx, int Ly, int Lz, atom_t* host_atoms, float* host_AA_, float* host_BB, float* host_transform_array, int max_atoms) {
    size_t atoms_size = Lx * Ly * Lz * sizeof(atom_t);
    size_t AA_size = Nconfig * 6 * sizeof(float);
    size_t BB_size = Nconfig * dir_number * 9 * sizeof(float);
    size_t transform_size = Nconfig * 6 * sizeof(float);
    max_count = max_atoms;
    cudaMalloc(&dev_atoms_read, atoms_size);
    cudaMalloc(&dev_atoms_write, atoms_size);
    cudaMalloc(&dev_AA_, AA_size);
    cudaMalloc(&dev_BB, BB_size);
    cudaMalloc(&dev_transform_array, transform_size);
    cudaMalloc(&dev_states, max_atoms * sizeof(curandState));
    cudaMemcpy(dev_atoms_read, host_atoms, atoms_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_atoms_write, host_atoms, atoms_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_AA_, host_AA_, AA_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_BB, host_BB, BB_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_transform_array, host_transform_array, transform_size, cudaMemcpyHostToDevice);
    dim3 block(128);
    dim3 grid((max_atoms + block.x - 1) / block.x);
    setup_kernel<<<grid, block>>>(dev_states, 19);
    cudaDeviceSynchronize();
}

extern "C" void cuda_cleanup() {
    cudaFree(dev_atoms_read);
    cudaFree(dev_atoms_write);
    cudaFree(dev_AA_);
    cudaFree(dev_BB);
    cudaFree(dev_transform_array);
    cudaFree(dev_states);
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

    cuda_sync_atoms(host_atoms, Lx, Ly, Lz);

    int *dev_xs, *dev_ys, *dev_zs;
    int *dev_ochered_count, *dev_ochered_x, *dev_ochered_y, *dev_ochered_z;
    curandState* dev_states;
    int max_ochered_size = count * (dir_number + 1); // Каждый атом + до dir_number соседей

    cudaMalloc(&dev_xs, count * sizeof(int));
    cudaMalloc(&dev_ys, count * sizeof(int));
    cudaMalloc(&dev_zs, count * sizeof(int));
    cudaMalloc(&dev_ochered_count, sizeof(int));
    cudaMalloc(&dev_ochered_x, max_ochered_size * sizeof(int));
    cudaMalloc(&dev_ochered_y, max_ochered_size * sizeof(int));
    cudaMalloc(&dev_ochered_z, max_ochered_size * sizeof(int));
    cudaMalloc(&dev_states, count * sizeof(curandState));

 
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

    dim3 setup_grid((count + 127) / 128);
    dim3 setup_block(128);
    setup_kernel<<<setup_grid, setup_block>>>(dev_states, 19);
    cudaDeviceSynchronize();

    int zero = 0;
    cudaMemcpy(dev_ochered_count, &zero, sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(128);
    dim3 grid((count + block.x - 1) / block.x);
    set_config_kernel<<<grid, block>>>(dev_atoms_write, Lx, Ly, Lz, dev_xs, dev_ys, dev_zs, count);
    cudaDeviceSynchronize();

    axyz_kernel<<<grid, block>>>(dev_atoms_read, dev_atoms_write, Lx, Ly, Lz, dev_xs, dev_ys, dev_zs, count, dev_states, T,
                                 dev_AA_, dev_BB, dev_transform_array,
                                 dev_ochered_x, dev_ochered_y, dev_ochered_z, dev_ochered_count, max_ochered_size);
    cudaDeviceSynchronize();

    // Копируем обновлённые атомы обратно на хост
    for (int i = 0; i < count; i++) {
        int x = atoms_to_update[i].x;
        int y = atoms_to_update[i].y;
        int z = atoms_to_update[i].z;
        int idx = get_atom_idx_host(x, y, z, Lx, Ly, Lz);
        cudaMemcpy(&host_atoms[idx], &dev_atoms_write[idx], sizeof(atom_t), cudaMemcpyDeviceToHost);
    }
    cudaMemcpy(dev_atoms_read, dev_atoms_write, Lx * Ly * Lz * sizeof(atom_t), cudaMemcpyDeviceToDevice);

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
    cudaFree(dev_xs);
    cudaFree(dev_ys);
    cudaFree(dev_zs);
    cudaFree(dev_ochered_count);
    cudaFree(dev_ochered_x);
    cudaFree(dev_ochered_y);
    cudaFree(dev_ochered_z);
    cudaFree(dev_states);
}

__global__ void verify_kernel(
    atom_t* atoms_cuda, atom_t* atoms_cpu,
    int Lx, int Ly, int Lz,
    float* max_diff, float* avg_diff)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Lx * Ly * Lz) return;
    
    float diff_x = fabsf(atoms_cuda[idx].a.x - atoms_cpu[idx].a.x);
    float diff_y = fabsf(atoms_cuda[idx].a.y - atoms_cpu[idx].a.y);
    float diff_z = fabsf(atoms_cuda[idx].a.z - atoms_cpu[idx].a.z);
    
    float max_val = max(diff_x, max(diff_y, diff_z));
    
    int* max_diff_int = (int*)max_diff;
    int max_val_int = __float_as_int(max_val);
    int old = *max_diff_int;
    while (max_val_int > old) {
        old = atomicCAS(max_diff_int, old, max_val_int);
    }
    
    atomicAdd(avg_diff, (diff_x + diff_y + diff_z) / 3.0f);
}