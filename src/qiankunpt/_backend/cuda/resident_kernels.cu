#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/complex.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
using Bits = unsigned long long;

__device__ __forceinline__ Bits splitmix64(Bits x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// One thread reads an edge once and updates every replica. The accumulator is
// complex amplitude, never squared here: cross-source/rank interference must
// survive until the all-reduce and final contraction.
template<bool POWER_OF_TWO>
__global__ void weighted_sketch_kernel(double* sketch, const int64_t* states,
        const c10::complex<double>* values, const double* denominators,
        const int64_t* seeds, int64_t size, int64_t replicas, Bits buckets) {
    for (int64_t edge = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         edge < size; edge += int64_t(gridDim.x) * blockDim.x) {
        const Bits key = static_cast<Bits>(states[edge]);
        const double d = denominators[edge];
        const double scale = sqrt(fabs(d));
        const double re = values[edge].real() / scale;
        const double im = values[edge].imag() / scale;
        const int sector = d < 0.0;
        for (int64_t r = 0; r < replicas; ++r) {
            const Bits mixed = splitmix64(key ^ static_cast<Bits>(seeds[r]));
            const Bits bucket = POWER_OF_TWO ? mixed & (buckets - 1) : mixed % buckets;
            double wr, wi;
            switch (mixed >> 62) {
                case 0: wr = re; wi = im; break;
                case 1: wr = -re; wi = -im; break;
                case 2: wr = -im; wi = re; break;
                default: wr = im; wi = -re;
            }
            double* dest = sketch + ((r * 2 + sector) * buckets + bucket) * 2;
            atomicAdd(dest, wr);
            atomicAdd(dest + 1, wi);
        }
    }
}

// Expose edge/replica pairs as independent threads. This keeps the same hash,
// phases and atomic sums, but trades repeated cached reads for more outstanding
// work when a thread's serial replica loop is limited by memory queues.
template<bool POWER_OF_TWO>
__global__ void weighted_sketch_replica_kernel(double* sketch, const int64_t* states,
        const c10::complex<double>* values, const double* denominators,
        const int64_t* seeds, int64_t size, int64_t replicas, Bits buckets) {
    for (int64_t item = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         item < size * replicas; item += int64_t(gridDim.x) * blockDim.x) {
        const int64_t edge = item / replicas, r = item % replicas;
        const Bits key = static_cast<Bits>(states[edge]);
        const double d = denominators[edge], scale = sqrt(fabs(d));
        const double re = values[edge].real() / scale, im = values[edge].imag() / scale;
        const Bits mixed = splitmix64(key ^ static_cast<Bits>(seeds[r]));
        const Bits bucket = POWER_OF_TWO ? mixed & (buckets - 1) : mixed % buckets;
        double wr, wi;
        switch (mixed >> 62) {
            case 0: wr = re; wi = im; break;
            case 1: wr = -re; wi = -im; break;
            case 2: wr = -im; wi = re; break;
            default: wr = im; wi = -re;
        }
        double* dest = sketch + ((r * 2 + (d < 0.0)) * buckets + bucket) * 2;
        atomicAdd(dest, wr); atomicAdd(dest + 1, wi);
    }
}

void vector_check(const at::Tensor& x, const at::Tensor& states, at::ScalarType type) {
    TORCH_CHECK(x.is_cuda() && x.device() == states.device() && x.is_contiguous()
                && x.dim() == 1 && x.scalar_type() == type,
                "native sketch requires contiguous vectors on the same CUDA device");
}
}

void weighted_sketch_launch(at::Tensor sketch, at::Tensor states, at::Tensor values,
                           at::Tensor denominator, at::Tensor seeds, int64_t buckets, bool replica_parallel) {
    vector_check(states, states, at::kLong);
    vector_check(values, states, at::kComplexDouble);
    vector_check(denominator, states, at::kDouble);
    vector_check(seeds, states, at::kLong);
    TORCH_CHECK(buckets > 0 && seeds.numel() > 0, "invalid sketch dimensions");
    TORCH_CHECK(values.numel() == states.numel() && denominator.numel() == states.numel(),
                "native sketch edge lengths differ");
    TORCH_CHECK(sketch.is_cuda() && sketch.device() == states.device() && sketch.is_contiguous()
                && sketch.scalar_type() == at::kDouble && sketch.dim() == 2
                && sketch.size(0) == seeds.numel() * 2 * buckets && sketch.size(1) == 2,
                "invalid native sketch accumulator");
    const c10::cuda::CUDAGuard guard(states.device());
    if (!states.numel()) return;
    const int64_t work = states.numel() * (replica_parallel ? seeds.numel() : 1);
    const int blocks = static_cast<int>(std::min<int64_t>((work + 255) / 256, 65535));
    const auto stream = c10::cuda::getCurrentCUDAStream();
    if (replica_parallel) {
        if ((buckets & (buckets - 1)) == 0)
            weighted_sketch_replica_kernel<true><<<blocks, 256, 0, stream>>>(sketch.data_ptr<double>(),
                states.data_ptr<int64_t>(), values.data_ptr<c10::complex<double>>(),
                denominator.data_ptr<double>(), seeds.data_ptr<int64_t>(), states.numel(), seeds.numel(), buckets);
        else
            weighted_sketch_replica_kernel<false><<<blocks, 256, 0, stream>>>(sketch.data_ptr<double>(),
                states.data_ptr<int64_t>(), values.data_ptr<c10::complex<double>>(),
                denominator.data_ptr<double>(), seeds.data_ptr<int64_t>(), states.numel(), seeds.numel(), buckets);
    } else if ((buckets & (buckets - 1)) == 0)
        weighted_sketch_kernel<true><<<blocks, 256, 0, stream>>>(sketch.data_ptr<double>(),
            states.data_ptr<int64_t>(), values.data_ptr<c10::complex<double>>(),
            denominator.data_ptr<double>(), seeds.data_ptr<int64_t>(), states.numel(), seeds.numel(), buckets);
    else
        weighted_sketch_kernel<false><<<blocks, 256, 0, stream>>>(sketch.data_ptr<double>(),
            states.data_ptr<int64_t>(), values.data_ptr<c10::complex<double>>(),
            denominator.data_ptr<double>(), seeds.data_ptr<int64_t>(), states.numel(), seeds.numel(), buckets);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void weighted_sketch_cuda(at::Tensor sketch, at::Tensor states, at::Tensor values,
                         at::Tensor denominator, at::Tensor seeds, int64_t buckets) {
    weighted_sketch_launch(sketch, states, values, denominator, seeds, buckets, false);
}

void weighted_sketch_replica_cuda(at::Tensor sketch, at::Tensor states, at::Tensor values,
                                 at::Tensor denominator, at::Tensor seeds, int64_t buckets) {
    weighted_sketch_launch(sketch, states, values, denominator, seeds, buckets, true);
}
