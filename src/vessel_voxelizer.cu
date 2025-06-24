#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

namespace nb = nanobind;
using namespace nb::literals;
#define K 10

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template <typename scalar>
__device__ scalar sqdist_point_linesegment(const scalar p[3], const scalar ls_start[3], const scalar ls_end[3]) {

    scalar ls_dir[3] = {ls_end[0] - ls_start[0], ls_end[1] - ls_start[1], ls_end[2] - ls_start[2]};
    scalar p_to_ls_start[3] = {p[0] - ls_start[0], p[1] - ls_start[1], p[2] - ls_start[2]};

    scalar ls_dir_dot = ls_dir[0] * ls_dir[0] + ls_dir[1] * ls_dir[1] + ls_dir[2] * ls_dir[2];
    scalar t = (p_to_ls_start[0] * ls_dir[0] + p_to_ls_start[1] * ls_dir[1] + p_to_ls_start[2] * ls_dir[2]) / ls_dir_dot;

    t = max(0.0f, min(1.0f, t));

    scalar p_proj[3] = {
        ls_start[0] + t * ls_dir[0],
        ls_start[1] + t * ls_dir[1],
        ls_start[2] + t * ls_dir[2]
    };

    scalar diff[3] = {
        p[0] - p_proj[0],
        p[1] - p_proj[1],
        p[2] - p_proj[2]
    };

    return diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
}


template<typename scalar>
__global__ void voxelize_kernel(scalar* volume,
                                scalar* volume_start,
                                scalar* vol_spacing,
                                int n_x, int n_y, int n_z,
                                const scalar* vessels,
                                const scalar* vessel_bbox,
                                const scalar* rads,
                                int n_vessels) {


    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (!(x < n_x && y < n_y && z < n_z)) // thread outside of voxel grid
        return;

    scalar spacing[3] = {vol_spacing[0], vol_spacing[1], vol_spacing[2]};

    scalar voxel_min[3] = {
        volume_start[0] + x * spacing[0],
        volume_start[1] + y * spacing[1],
        volume_start[2] + z * spacing[2]
    };
    scalar voxel_max[3] = {
        voxel_min[0] + spacing[0],
        voxel_min[1] + spacing[1],
        voxel_min[2] + spacing[2]
    };

    uint8_t internal_point[K * K * K] = {};
    scalar check_point[3];
    scalar ls_start[3];
    scalar ls_end[3];
    scalar radius;
    scalar dist_squared;

    for (int i = 0; i < n_vessels; ++i) {

        bool intersects_bbox =
            (voxel_max[0] >= vessel_bbox[0 + 0 * 3 + 2 * 3 * i] && voxel_min[0] <= vessel_bbox[0 + 1 * 3 + 2 * 3 * i]) &&
            (voxel_max[1] >= vessel_bbox[1 + 0 * 3 + 2 * 3 * i] && voxel_min[1] <= vessel_bbox[1 + 1 * 3 + 2 * 3 * i]) &&
            (voxel_max[2] >= vessel_bbox[2 + 0 * 3 + 2 * 3 * i] && voxel_min[2] <= vessel_bbox[2 + 1 * 3 + 2 * 3 * i]);

        if (!intersects_bbox) {
            continue; // Skip this vessel if it doesn't intersect the voxel
        }

         // Further intersection test with the vessel line segment
        ls_start[0] = vessels[0 + 0 * 3 + 2 * 3 * i];
        ls_start[1] = vessels[1 + 0 * 3 + 2 * 3 * i];
        ls_start[2] = vessels[2 + 0 * 3 + 2 * 3 * i];
        ls_end[0] = vessels[0 + 1 * 3 + 2 * 3 * i];
        ls_end[1] = vessels[1 + 1 * 3 + 2 * 3 * i];
        ls_end[2] = vessels[2 + 1 * 3 + 2 * 3 * i];
        radius = rads[i];

        // Check a KxKxK grid within the voxel
        for (int ix = 0; ix < K; ++ix) {
           for (int iy = 0; iy < K; ++iy) {
               for (int iz = 0; iz < K; ++iz) {
                   check_point[0] = voxel_min[0] + (1.0 / (2 * K) + (1.0 * ix / K)) * spacing[0];
                   check_point[1] = voxel_min[1] + (1.0 / (2 * K) + (1.0 * iy / K)) * spacing[1];
                   check_point[2] = voxel_min[2] + (1.0 / (2 * K) + (1.0 * iz / K)) * spacing[2];

                   dist_squared = sqdist_point_linesegment(check_point, ls_start, ls_end);
                   if (dist_squared <= (radius * radius)) {
                       internal_point[(iz * K * K) + (iy * K) + ix] = 1;
                   }
               }
           }
        }

        int intersections = 0;
        for (int i = 0; i < (K * K * K); i++)
            intersections += internal_point[i];

        volume[z + n_z * y + n_z * n_y * x] = (scalar) intersections / (K * K * K);
    }

}

template<typename scalar>
void voxelize(nb::ndarray<scalar, nb::ndim<3>, nb::c_contig, nb::device::cpu> volume,
              nb::ndarray<scalar, nb::shape<3>, nb::c_contig, nb::device::cpu> volume_start,
              nb::ndarray<scalar, nb::shape<3>, nb::c_contig, nb::device::cpu> vol_spacing,
              nb::ndarray<scalar, nb::shape<-1, 2, 3>, nb::c_contig, nb::device::cpu> vessels,
              nb::ndarray<scalar, nb::shape<-1, 2, 3>, nb::c_contig, nb::device::cpu> vessel_bbox,
              nb::ndarray<scalar, nb::shape<-1>, nb::c_contig, nb::device::cpu> rads) {

    // --- 1. Get dimensions from input arrays ---
    size_t n_x = volume.shape(0);
    size_t n_y = volume.shape(1);
    size_t n_z = volume.shape(2);
    size_t n_vessels = vessels.shape(0);
    size_t vol_size_bytes = n_x * n_y * n_z * sizeof(scalar);
    size_t vec3_size_bytes = 3 * sizeof(scalar);
    size_t vessels_size_bytes = n_vessels * 2 * 3 * sizeof(scalar);
    size_t rads_size_bytes = n_vessels * sizeof(scalar);


    // --- 2. Configure kernel launch parameters ---
    const dim3 threads(8, 8, 8);
    const dim3 blocks((n_x + threads.x - 1) / threads.x,
                      (n_y + threads.y - 1) / threads.y,
                      (n_z + threads.z - 1) / threads.z);

    std::cout << "---------- Vessel Voxelizer -----------" << std::endl;
    std::cout << "Volume: " << n_x << "x" << n_y << "x"<< n_z << std::endl;
    std::cout << "Vessels: " << n_vessels << std::endl;
    std::cout << "Grid Dim: " << blocks.x << ", " << blocks.y << ", "<< blocks.z << std::endl;
    std::cout << "Block Dim: " << threads.x << ", " << threads.y << ", "<< threads.z << std::endl;
    std::cout << "---------------------------------------" << std::endl;

    // --- 3. Allocate memory on the GPU ---
    scalar *d_volume, *d_volume_start, *d_vol_spacing, *d_vessels, *d_vessel_bbox, *d_rads;
    gpuErrchk(cudaMalloc((void**)&d_volume, vol_size_bytes));
    gpuErrchk(cudaMalloc((void**)&d_volume_start, vec3_size_bytes));
    gpuErrchk(cudaMalloc((void**)&d_vol_spacing, vec3_size_bytes));
    gpuErrchk(cudaMalloc((void**)&d_vessels, vessels_size_bytes));
    gpuErrchk(cudaMalloc((void**)&d_vessel_bbox, vessels_size_bytes));
    gpuErrchk(cudaMalloc((void**)&d_rads, rads_size_bytes));

    std::chrono::steady_clock::time_point time_start = std::chrono::steady_clock::now();

    // --- 4. Copy data from Host (CPU) to Device (GPU) ---
    gpuErrchk(cudaMemcpy(d_volume, volume.data(), vol_size_bytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_volume_start, volume_start.data(), vec3_size_bytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_vol_spacing, vol_spacing.data(), vec3_size_bytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_vessels, vessels.data(), vessels_size_bytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_vessel_bbox, vessel_bbox.data(), vessels_size_bytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_rads, rads.data(), rads_size_bytes, cudaMemcpyHostToDevice));

    std::chrono::steady_clock::time_point time_copied_to_gpu = std::chrono::steady_clock::now();

    // --- 5. Launch the CUDA kernel ---
    voxelize_kernel<scalar><<<blocks, threads>>>(d_volume,
                                                 d_volume_start,
                                                 d_vol_spacing,
                                                 n_x, n_y, n_z,
                                                 d_vessels,
                                                 d_vessel_bbox,
                                                 d_rads,
                                                 n_vessels);

    // Check for any errors launched from the kernel
    gpuErrchk(cudaPeekAtLastError());
    // Block until the device has completed all preceding tasks
    gpuErrchk(cudaDeviceSynchronize());

    std::chrono::steady_clock::time_point time_kernel_end = std::chrono::steady_clock::now();

    // --- 6. Copy result data from Device (GPU) to Host (CPU) ---
    gpuErrchk(cudaMemcpy(volume.data(), d_volume, vol_size_bytes, cudaMemcpyDeviceToHost));

    std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();

    // --- 7. Free allocated GPU memory ---
    gpuErrchk(cudaFree(d_volume));
    gpuErrchk(cudaFree(d_volume_start));
    gpuErrchk(cudaFree(d_vol_spacing));
    gpuErrchk(cudaFree(d_vessels));
    gpuErrchk(cudaFree(d_vessel_bbox));
    gpuErrchk(cudaFree(d_rads));

    // --- 8. Print timing information ---
    float h2d_ms = (float) std::chrono::duration_cast<std::chrono::microseconds>(time_copied_to_gpu - time_start).count() / 1000;
    float kernel_ms = (float) std::chrono::duration_cast<std::chrono::microseconds>(time_kernel_end - time_copied_to_gpu).count() / 1000;
    float d2h_ms = (float) std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_kernel_end).count() / 1000;
    float total_ms = (float) std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / 1000;

    std::cout << "Finished! " << std::endl;
    std::cout << "  - HtoD Transfer Time: " << h2d_ms << " ms" << std::endl;
    std::cout << "  - Kernel Time:        " << kernel_ms << " ms" << std::endl;
    std::cout << "  - DtoH Transfer Time: " << d2h_ms << " ms" << std::endl;
    std::cout << "  - Total Time:         " << total_ms << " ms" << std::endl;
}


// --- Nanobind module definition ---
NB_MODULE(_vessel_vox, m) {
    m.doc() = "A CUDA-accelerated vessel voxelizer for Python.";
    m.def("__voxelize", &voxelize<float>,
        "volume"_a.noconvert(),
        "volume_start"_a.noconvert(),
        "vol_spacing"_a.noconvert(),
        "vessels"_a.noconvert(),
        "vessel_bbox"_a.noconvert(),
        "rads"_a.noconvert(),
        "Performs vessel voxelization on the GPU."
    );
}