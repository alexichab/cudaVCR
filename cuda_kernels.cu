#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "mc_growth.h"

// Генерация случайных смещений на GPU
__device__ void random_displacements_gpu(float* ax_, float* ay_, float* az_, unsigned short int config_, float* transform_array, curandState* state, float T) {
    if (config_ >= Nconfig || config_ == 65535) {
        *ax_ = 0.0f; *ay_ = 0.0f; *az_ = 0.0f;
        return;
    }
    if (T <= 0) {
        *ax_ = 0.0f; *ay_ = 0.0f; *az_ = 0.0f;
        return;
    }
    float a1 = curand_normal(state);
    float a2 = curand_normal(state);
    float a3 = curand_normal(state);
    float coeff = sqrtf(0.5f * T);
    float* p = transform_array + config_ * 6;
    *ax_ = coeff * (p[0] * a1 + p[3] * a2 + p[4] * a3);
    *ay_ = coeff * (p[3] * a1 + p[1] * a2 + p[5] * a3);
    *az_ = coeff * (p[4] * a1 + p[5] * a2 + p[2] * a3);
}

// Реализация calc_x2y2z2 для CUDA
__device__ void calc_x2y2z2(int x, int y, int z, int dir, int Lx, int Ly, int Lz, int* x2, int* y2, int* z2) {
    if (dir < 0 || dir >= dir_number) {
        printf("calc_x2y2z2: invalid dir=%d, dir_number=%d\n", dir, dir_number);
        *x2 = x; *y2 = y; *z2 = z;
        return;
    }
    int factor = (z % 2 == 0) ? 1 : -1;
    // Расширено до 16 направлений
    const int dx[16] = {1, 1, -1, -1, 0, 2, 2, 0, 2, 2, 0, -2, -2, 0, -2, -2};
    const int dy[16] = {1, -1, 1, -1, 2, 0, 2, -2, 0, -2, 2, 0, 2, -2, 0, -2};
    const int dz[16] = {1, -1, -1, 1, 2, 2, 0, -2, -2, 0, -2, -2, 0, 2, 2, 0};
    
    *x2 = x + factor * dx[dir];
    *y2 = y + factor * dy[dir];
    *z2 = z + factor * dz[dir];
    
    if (*x2 < 0) *x2 += Lx;
    if (*x2 >= Lx) *x2 -= Lx;
    if (*y2 < 0) *y2 += Ly;
    if (*y2 >= Ly) *y2 -= Ly;
    *z2 = max(2, min(Lz - 3, *z2));
}

// Ядро для обновления смещений атомов
__global__ void axyz_kernel(
    atom_t* atoms,
    int Lx, int Ly, int Lz,
    int* xs, int* ys, int* zs,
    int count,
    curandState* states,
    float T,
    float* d_AA_,
    float* d_BB,
    float* d_transform_array,
    int* d_ochered_count,
    int* d_ochered_x,
    int* d_ochered_y,
    int* d_ochered_z,
    int max_ochered_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    int x = xs[idx];
    int y = ys[idx];
    int z = zs[idx];
    // Проверка валидности координат
    if (x < 0 || x >= Lx || y < 0 || y >= Ly || z < 0 || z >= Lz) {
        printf("axyz_kernel: invalid coordinates: idx=%d, x=%d, y=%d, z=%d\n", idx, x, y, z);
        return;
    }
    int atom_idx = z * Lx * Ly + y * Lx + x;
    if (atom_idx < 0 || atom_idx >= Lx * Ly * Lz) {
        printf("axyz_kernel: invalid atom_idx=%d for x=%d, y=%d, z=%d\n", atom_idx, x, y, z);
        return;
    }

    if (atoms[atom_idx].type == 0) return;

    unsigned short config_ = atoms[atom_idx].config;
    if (config_ >= Nconfig || config_ == 65535) {
        printf("axyz_kernel: invalid config_=%u at atom_idx=%d\n", config_, atom_idx);
        return;
    }

    float* AA_ptr = d_AA_ + config_ * 6;
    float A_xx = AA_ptr[0];
    float A_yy = AA_ptr[1];
    float A_zz = AA_ptr[2];
    float A_xy = AA_ptr[3];
    float A_xz = AA_ptr[4];
    float A_yz = AA_ptr[5];

    float Bx = atoms[atom_idx].B0.x;
    float By = atoms[atom_idx].B0.y;
    float Bz = atoms[atom_idx].B0.z;

    for (int dir = 0; dir < dir_number; dir++) {
        int x2, y2, z2;
        calc_x2y2z2(x, y, z, dir, Lx, Ly, Lz, &x2, &y2, &z2);
        int neighbor_idx = z2 * Lx * Ly + y2 * Lx + x2;

        if (neighbor_idx >= 0 && neighbor_idx < Lx * Ly * Lz) {
            float ax2 = atoms[neighbor_idx].a.x;
            float ay2 = atoms[neighbor_idx].a.y;
            float az2 = atoms[neighbor_idx].a.z;

            float* BB_ptr = d_BB + config_ * dir_number * 9 + dir * 9;
            float Bxx = BB_ptr[0]; float Bxy = BB_ptr[1]; float Bxz = BB_ptr[2];
            float Byx = BB_ptr[3]; float Byy = BB_ptr[4]; float Byz = BB_ptr[5];
            float Bzx = BB_ptr[6]; float Bzy = BB_ptr[7]; float Bzz = BB_ptr[8];

            Bx += Bxx * ax2 + Bxy * ay2 + Bxz * az2;
            By += Byx * ax2 + Byy * ay2 + Byz * az2;
            Bz += Bzx * ax2 + Bzy * ay2 + Bzz * az2;

            int ochered_idx = atomicAdd(d_ochered_count, 1);
            if (ochered_idx < max_ochered_size) {
                d_ochered_x[ochered_idx] = x2;
                d_ochered_y[ochered_idx] = y2;
                d_ochered_z[ochered_idx] = z2;
            } else {
                printf("axyz_kernel: ochered overflow: ochered_idx=%d, max_ochered_size=%d\n", ochered_idx, max_ochered_size);
            }
        }
    }

    int ochered_idx = atomicAdd(d_ochered_count, 1);
    if (ochered_idx < max_ochered_size) {
        d_ochered_x[ochered_idx] = x;
        d_ochered_y[ochered_idx] = y;
        d_ochered_z[ochered_idx] = z;
    } else {
        printf("axyz_kernel: ochered overflow for self: ochered_idx=%d, max_ochered_size=%d\n", ochered_idx, max_ochered_size);
    }

    curandState local_state = states[idx];
    float ax_, ay_, az_;
    random_displacements_gpu(&ax_, &ay_, &az_, config_, d_transform_array, &local_state, T);
    states[idx] = local_state;

    ax_ -= 0.5f * (A_xx * Bx + A_xy * By + A_xz * Bz);
    ay_ -= 0.5f * (A_xy * Bx + A_yy * By + A_yz * Bz);
    az_ -= 0.5f * (A_xz * Bx + A_yz * By + A_zz * Bz);

    atoms[atom_idx].a.x = ax_;
    atoms[atom_idx].a.y = ay_;
    atoms[atom_idx].a.z = az_;
}

// Инициализация генераторов случайных чисел
__global__ void setup_kernel(curandState* state, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

// Внешняя функция для вызова из C++
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
    // Проверка входных параметров
    if (!atoms_to_update || !host_atoms || !host_AA_ || !host_BB || !host_transform_array) {
        printf("cuda_do_many_axyz: null pointer detected: atoms_to_update=%p, host_atoms=%p, host_AA_=%p, host_BB=%p, host_transform_array=%p\n",
               atoms_to_update, host_atoms, host_AA_, host_BB, host_transform_array);
        exit(1);
    }
    if (count <= 0 || Lx <= 0 || Ly <= 0 || Lz <= 0) {
        printf("cuda_do_many_axyz: invalid parameters: count=%d, Lx=%d, Ly=%d, Lz=%d\n", count, Lx, Ly, Lz);
        exit(1);
    }
    printf("cuda_do_many_axyz: starting with count=%d, Lx=%d, Ly=%d, Lz=%d, dir_number=%d\n", count, Lx, Ly, Lz, dir_number);

    // Инициализация событий для замера времени
    cudaEvent_t start, stop, memcpy_h2d, memcpy_d2h, kernel_setup, kernel_axyz;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&memcpy_h2d);
    cudaEventCreate(&memcpy_d2h);
    cudaEventCreate(&kernel_setup);
    cudaEventCreate(&kernel_axyz);

    // Начало общего замера
    cudaEventRecord(start);

    atom_t* dev_atoms;
    int *dev_xs, *dev_ys, *dev_zs;
    curandState* dev_states;
    float *dev_AA_, *dev_BB, *dev_transform_array;
    int *dev_ochered_count, *dev_ochered_x, *dev_ochered_y, *dev_ochered_z;

    size_t atoms_size = Lx * Ly * Lz * sizeof(atom_t);
    size_t AA_size = Nconfig * 6 * sizeof(float);
    size_t BB_size = Nconfig * dir_number * 9 * sizeof(float); // Исправлено для dir_number
    size_t transform_size = Nconfig * 6 * sizeof(float);
    int max_ochered_size = count * (dir_number + 1); // Учитываем всех соседей + сам атом

    // Выделение памяти с проверками
    if (cudaMalloc(&dev_atoms, atoms_size) != cudaSuccess) {
        printf("cuda_do_many_axyz: cudaMalloc failed for dev_atoms\n");
        exit(1);
    }
    if (cudaMalloc(&dev_xs, count * sizeof(int)) != cudaSuccess) {
        printf("cuda_do_many_axyz: cudaMalloc failed for dev_xs\n");
        exit(1);
    }
    if (cudaMalloc(&dev_ys, count * sizeof(int)) != cudaSuccess) {
        printf("cuda_do_many_axyz: cudaMalloc failed for dev_ys\n");
        exit(1);
    }
    if (cudaMalloc(&dev_zs, count * sizeof(int)) != cudaSuccess) {
        printf("cuda_do_many_axyz: cudaMalloc failed for dev_zs\n");
        exit(1);
    }
    if (cudaMalloc(&dev_states, count * sizeof(curandState)) != cudaSuccess) {
        printf("cuda_do_many_axyz: cudaMalloc failed for dev_states\n");
        exit(1);
    }
    if (cudaMalloc(&dev_AA_, AA_size) != cudaSuccess) {
        printf("cuda_do_many_axyz: cudaMalloc failed for dev_AA_\n");
        exit(1);
    }
    if (cudaMalloc(&dev_BB, BB_size) != cudaSuccess) {
        printf("cuda_do_many_axyz: cudaMalloc failed for dev_BB\n");
        exit(1);
    }
    if (cudaMalloc(&dev_transform_array, transform_size) != cudaSuccess) {
        printf("cuda_do_many_axyz: cudaMalloc failed for dev_transform_array\n");
        exit(1);
    }
    if (cudaMalloc(&dev_ochered_count, sizeof(int)) != cudaSuccess) {
        printf("cuda_do_many_axyz: cudaMalloc failed for dev_ochered_count\n");
        exit(1);
    }
    if (cudaMalloc(&dev_ochered_x, max_ochered_size * sizeof(int)) != cudaSuccess) {
        printf("cuda_do_many_axyz: cudaMalloc failed for dev_ochered_x\n");
        exit(1);
    }
    if (cudaMalloc(&dev_ochered_y, max_ochered_size * sizeof(int)) != cudaSuccess) {
        printf("cuda_do_many_axyz: cudaMalloc failed for dev_ochered_y\n");
        exit(1);
    }
    if (cudaMalloc(&dev_ochered_z, max_ochered_size * sizeof(int)) != cudaSuccess) {
        printf("cuda_do_many_axyz: cudaMalloc failed for dev_ochered_z\n");
        exit(1);
    }

    // Замер копирования Host-to-Device
    cudaEventRecord(memcpy_h2d);
    if (cudaMemcpy(dev_atoms, host_atoms, atoms_size, cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("cuda_do_many_axyz: cudaMemcpy failed for dev_atoms\n");
        exit(1);
    }
    if (cudaMemcpy(dev_AA_, host_AA_, AA_size, cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("cuda_do_many_axyz: cudaMemcpy failed for dev_AA_\n");
        exit(1);
    }
    if (cudaMemcpy(dev_BB, host_BB, BB_size, cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("cuda_do_many_axyz: cudaMemcpy failed for dev_BB\n");
        exit(1);
    }
    if (cudaMemcpy(dev_transform_array, host_transform_array, transform_size, cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("cuda_do_many_axyz: cudaMemcpy failed for dev_transform_array\n");
        exit(1);
    }

    int* host_xs = new int[count];
    int* host_ys = new int[count];
    int* host_zs = new int[count];
    if (!host_xs || !host_ys || !host_zs) {
        printf("cuda_do_many_axyz: failed to allocate host_xs, host_ys, or host_zs\n");
        exit(1);
    }
    for (int i = 0; i < count; i++) {
        host_xs[i] = atoms_to_update[i].x;
        host_ys[i] = atoms_to_update[i].y;
        host_zs[i] = atoms_to_update[i].z;
        // Проверка координат на CPU
        if (host_xs[i] < 0 || host_xs[i] >= Lx || host_ys[i] < 0 || host_ys[i] >= Ly || host_zs[i] < 0 || host_zs[i] >= Lz) {
            printf("cuda_do_many_axyz: invalid coordinates at index %d: x=%d, y=%d, z=%d\n",
                   i, host_xs[i], host_ys[i], host_zs[i]);
            delete[] host_xs; delete[] host_ys; delete[] host_zs;
            exit(1);
        }
    }
    if (cudaMemcpy(dev_xs, host_xs, count * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("cuda_do_many_axyz: cudaMemcpy failed for dev_xs\n");
        exit(1);
    }
    if (cudaMemcpy(dev_ys, host_ys, count * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("cuda_do_many_axyz: cudaMemcpy failed for dev_ys\n");
        exit(1);
    }
    if (cudaMemcpy(dev_zs, host_zs, count * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("cuda_do_many_axyz: cudaMemcpy failed for dev_zs\n");
        exit(1);
    }
    delete[] host_xs;
    delete[] host_ys;
    delete[] host_zs;

    int zero = 0;
    if (cudaMemcpy(dev_ochered_count, &zero, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("cuda_do_many_axyz: cudaMemcpy failed for dev_ochered_count\n");
        exit(1);
    }
    cudaEventRecord(kernel_setup);

    // Запуск ядра инициализации генераторов случайных чисел
    dim3 block(128);
    dim3 grid((count + block.x - 1) / block.x);
    setup_kernel<<<grid, block>>>(dev_states, time(NULL));
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("cuda_do_many_axyz: setup_kernel failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // Замер выполнения axyz_kernel
    cudaEventRecord(kernel_axyz);
    axyz_kernel<<<grid, block>>>(dev_atoms, Lx, Ly, Lz, dev_xs, dev_ys, dev_zs, count, dev_states, T,
                                 dev_AA_, dev_BB, dev_transform_array,
                                 dev_ochered_count, dev_ochered_x, dev_ochered_y, dev_ochered_z, max_ochered_size);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("cuda_do_many_axyz: axyz_kernel failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // Замер копирования Device-to-Host
    cudaEventRecord(memcpy_d2h);
    if (cudaMemcpy(host_atoms, dev_atoms, atoms_size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("cuda_do_many_axyz: cudaMemcpy failed for host_atoms\n");
        exit(1);
    }
    int ochered_count;
    if (cudaMemcpy(&ochered_count, dev_ochered_count, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("cuda_do_many_axyz: cudaMemcpy failed for ochered_count\n");
        exit(1);
    }
    int* host_ochered_x = new int[max_ochered_size];
    int* host_ochered_y = new int[max_ochered_size];
    int* host_ochered_z = new int[max_ochered_size];
    if (!host_ochered_x || !host_ochered_y || !host_ochered_z) {
        printf("cuda_do_many_axyz: failed to allocate host_ochered_x, host_ochered_y, or host_ochered_z\n");
        exit(1);
    }
    if (cudaMemcpy(host_ochered_x, dev_ochered_x, max_ochered_size * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("cuda_do_many_axyz: cudaMemcpy failed for host_ochered_x\n");
        exit(1);
    }
    if (cudaMemcpy(host_ochered_y, dev_ochered_y, max_ochered_size * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("cuda_do_many_axyz: cudaMemcpy failed for host_ochered_y\n");
        exit(1);
    }
    if (cudaMemcpy(host_ochered_z, dev_ochered_z, max_ochered_size * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("cuda_do_many_axyz: cudaMemcpy failed for host_ochered_z\n");
        exit(1);
    }

    for (int i = 0; i < min(ochered_count, max_ochered_size); i++) {
        v_ochered_Edef(host_ochered_x[i], host_ochered_y[i], host_ochered_z[i]);
    }

    delete[] host_ochered_x;
    delete[] host_ochered_y;
    delete[] host_ochered_z;

    // Освобождение памяти
    cudaFree(dev_atoms);
    cudaFree(dev_xs);
    cudaFree(dev_ys);
    cudaFree(dev_zs);
    cudaFree(dev_states);
    cudaFree(dev_AA_);
    cudaFree(dev_BB);
    cudaFree(dev_transform_array);
    cudaFree(dev_ochered_count);
    cudaFree(dev_ochered_x);
    cudaFree(dev_ochered_y);
    cudaFree(dev_ochered_z);

    // Конец замера
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Вычисление времени
    float total_time, memcpy_h2d_time, memcpy_d2h_time, kernel_setup_time, kernel_axyz_time;
    cudaEventElapsedTime(&total_time, start, stop);
    cudaEventElapsedTime(&memcpy_h2d_time, memcpy_h2d, kernel_setup);
    cudaEventElapsedTime(&kernel_setup_time, kernel_setup, kernel_axyz);
    cudaEventElapsedTime(&kernel_axyz_time, kernel_axyz, memcpy_d2h);
    cudaEventElapsedTime(&memcpy_d2h_time, memcpy_d2h, stop);

    // Вывод результатов
    printf("cuda_do_many_axyz (count=%d):\n", count);
    printf("  Total time: %.3f ms\n", total_time);
    printf("  Memcpy Host-to-Device: %.3f ms (%.1f%%)\n", memcpy_h2d_time, 100.0f * memcpy_h2d_time / total_time);
    printf("  Setup kernel: %.3f ms (%.1f%%)\n", kernel_setup_time, 100.0f * kernel_setup_time / total_time);
    printf("  Axyz kernel: %.3f ms (%.1f%%)\n", kernel_axyz_time, 100.0f * kernel_axyz_time / total_time);
    printf("  Memcpy Device-to-Host: %.3f ms (%.1f%%)\n", memcpy_d2h_time, 100.0f * memcpy_d2h_time / total_time);

    // Проверка ошибок CUDA
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("cuda_do_many_axyz: CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // Очистка событий
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(memcpy_h2d);
    cudaEventDestroy(memcpy_d2h);
    cudaEventDestroy(kernel_setup);
    cudaEventDestroy(kernel_axyz);
}