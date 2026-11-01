#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <vector>
#include "screening.cuh"

namespace {
constexpr int THREADS = 128;
using Bits = unsigned long long;

struct Tables {
    const int64_t *so, *d_o;
    const int32_t *sa, *da;
    const double *sv, *dv, *h, *j;
    int n;
};

__device__ int parity(Bits state, int i, int a) {
    const int lo = min(i, a), hi = max(i, a);
    const Bits mask = ((Bits(1) << hi) - 1) ^ ((Bits(1) << lo) - 1);
    return __popcll(state & mask) & 1;
}

__device__ double diagonal_value(Bits state, Tables t, double core) {
    double value = 0.0;
    while (state) {
        const int i = __ffsll(state)-1;
        state &= state-1;
        value += t.h[i];
        Bits rest = state;
        while (rest) {
            const int k = __ffsll(rest)-1;
            rest &= rest-1;
            value += t.j[i*t.n+k];
        }
    }
    return value + core;
}

// One block owns a source row. Threads traverse only the sparse excitation
// lists belonging to occupied holes. Count -> device prefix sum -> fill avoids
// an O(M L^4) dense candidate buffer and any atomic append/overflow truncation.
template<bool WRITE, bool DIAGONALS, bool WARP = false>
__global__ void coupled_kernel(const int64_t* states, Tables t, double threshold,
                               double core, int64_t* counts, const int64_t* starts,
                               int64_t* targets, double* values, double* diagonals) {
    __shared__ int occupied[64], nocc;
    __shared__ double fields[64], source_diag;
    const int row = blockIdx.x, lane = threadIdx.x;
    // Warp layout gives different hole pairs to the four warps, avoiding a
    // block-wide serial traversal of every short sparse excitation list.
    constexpr int GROUP = WARP ? 32 : THREADS;
    constexpr int GROUPS = THREADS / GROUP;
    const int group = lane / GROUP, member = lane % GROUP;
    const Bits source = static_cast<Bits>(states[row]);
    if (lane == 0) {
        Bits rest = source;
        int k = 0;
        while (rest) {
            occupied[k++] = __ffsll(rest)-1;
            rest &= rest-1;
        }
        nocc = k;
        if constexpr (WRITE) source_diag = diagonal_value(source, t, core);
    }
    __syncthreads();
    if constexpr (WRITE && DIAGONALS) {
        if (lane < t.n) {
            double f = t.h[lane];
            for (int k = 0; k < nocc; ++k) f += t.j[lane*t.n+occupied[k]];
            fields[lane] = f;
        }
        __syncthreads();
    }
    int64_t cursor = 0;
    if constexpr (WRITE) cursor = starts[int64_t(row)*THREADS+lane];
    const int64_t initial = cursor;
    if (lane == 0) {
        if constexpr (WRITE) {
            targets[cursor] = static_cast<int64_t>(source);
            values[cursor] = source_diag;
            if constexpr (DIAGONALS) diagonals[cursor] = source_diag;
        }
        ++cursor;
    }
    for (int oi = 0; oi < nocc; ++oi) {
        const int i = occupied[oi];
        for (int64_t e = (oi % GROUPS == group ? t.so[i]+member : t.so[i+1]);
             e < t.so[i+1]; e += GROUP) {
            const int a = t.sa[e];
            if (source & (Bits(1) << a)) continue;
            const double* integrals = t.sv + e*(t.n+1);
            double v = integrals[0];
            for (int k = 0; k < nocc; ++k) v += integrals[1+occupied[k]];
            if (!keep_pt_coupling(v, threshold)) continue;
            if constexpr (WRITE) {
                Bits target = source ^ (Bits(1) << i);
                if (parity(target, i, a)) v = -v;
                target |= Bits(1) << a;
                targets[cursor] = static_cast<int64_t>(target);
                values[cursor] = v;
                if constexpr (DIAGONALS)
                    diagonals[cursor] = source_diag-fields[i]+fields[a]-t.j[i*t.n+a];
            }
            ++cursor;
        }
        for (int oj = oi+1+group; oj < nocc; oj += GROUPS) {
            const int j = occupied[oj], ij = i*t.n+j;
            for (int64_t e = t.d_o[ij]+member; e < t.d_o[ij+1]; e += GROUP) {
                const int a = t.da[2*e], b = t.da[2*e+1];
                if (source & ((Bits(1) << a) | (Bits(1) << b))) continue;
                double v = t.dv[e];
                if (!keep_pt_coupling(v, threshold)) continue;
                if constexpr (WRITE) {
                    Bits target = source ^ (Bits(1) << i);
                    const int sign1 = parity(target, i, a);
                    target |= Bits(1) << a;
                    target ^= Bits(1) << j;
                    const int sign2 = parity(target, j, b);
                    target |= Bits(1) << b;
                    if (sign1 ^ sign2) v = -v;
                    targets[cursor] = static_cast<int64_t>(target);
                    values[cursor] = v;
                    if constexpr (DIAGONALS)
                        diagonals[cursor] = source_diag-fields[i]-fields[j]+fields[a]+fields[b]
                            +t.j[i*t.n+j]+t.j[a*t.n+b]-t.j[i*t.n+a]-t.j[i*t.n+b]
                            -t.j[j*t.n+a]-t.j[j*t.n+b];
                }
                ++cursor;
            }
        }
    }
    if constexpr (!WRITE) counts[int64_t(row)*THREADS+lane] = cursor-initial;
}

__global__ void diagonal_kernel(const int64_t* states, int64_t size, Tables t,
                                double core, double* out) {
    for (int64_t row = int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
         row < size; row += int64_t(gridDim.x)*blockDim.x)
        out[row] = diagonal_value(static_cast<Bits>(states[row]), t, core);
}

__global__ void diagonal_shared_kernel(const int64_t* states, int64_t size, Tables t,
                                      double core, double* out) {
    extern __shared__ double cache[];
    for (int k = threadIdx.x; k < t.n; k += blockDim.x) cache[k] = t.h[k];
    for (int k = threadIdx.x; k < t.n*t.n; k += blockDim.x) cache[t.n+k] = t.j[k];
    __syncthreads();
    t.h = cache; t.j = cache+t.n;
    // Retain canonical target-only order, including every floating point add.
    for (int64_t row = int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
         row < size; row += int64_t(gridDim.x)*blockDim.x)
        out[row] = diagonal_value(static_cast<Bits>(states[row]), t, core);
}

void check(const at::Tensor& x, const at::Tensor& states, at::ScalarType dtype) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 1,
                "CUDA coupling expects contiguous one-dimensional tensors");
    TORCH_CHECK(x.device() == states.device() && x.scalar_type() == dtype,
                "CUDA coupling tensor device/dtype mismatch");
}

void check_states(const at::Tensor& states, int n) {
    check(states, states, at::kLong);
    TORCH_CHECK(n > 0 && n <= 64, "CUDA coupling supports 1..64 spin orbitals");
    TORCH_CHECK(states.numel() <= 2147483647, "Too many source rows for CUDA grid");
}

Tables table_view(const std::vector<at::Tensor>& ts, const at::Tensor& states, int n) {
    TORCH_CHECK(ts.size() == 8, "Expected eight Hamiltonian tables");
    const at::ScalarType types[] = {at::kLong, at::kInt, at::kDouble, at::kLong,
                                    at::kInt, at::kDouble, at::kDouble, at::kDouble};
    for (int k = 0; k < 8; ++k) check(ts[k], states, types[k]);
    TORCH_CHECK(ts[0].numel() == n+1 && ts[3].numel() == n*n+1 &&
                ts[2].numel() == ts[1].numel()*(n+1) &&
                ts[4].numel() == ts[5].numel()*2 &&
                ts[6].numel() == n && ts[7].numel() == n*n, "Invalid Hamiltonian table shapes");
    return {ts[0].data_ptr<int64_t>(), ts[3].data_ptr<int64_t>(),
            ts[1].data_ptr<int32_t>(), ts[4].data_ptr<int32_t>(),
            ts[2].data_ptr<double>(), ts[5].data_ptr<double>(),
            ts[6].data_ptr<double>(), ts[7].data_ptr<double>(), n};
}
}

template<bool WARP>
at::Tensor coupled_counts_impl(at::Tensor states, std::vector<at::Tensor> ts,
                               int n, double threshold) {
    check_states(states, n);
    TORCH_CHECK(std::isfinite(threshold) && threshold >= 0, "Invalid screening threshold");
    const c10::cuda::CUDAGuard guard(states.device());
    const auto t = table_view(ts, states, n);
    auto counts = at::empty({states.numel()*THREADS}, states.options());
    if (states.numel()) {
        coupled_kernel<false, false, WARP><<<states.numel(), THREADS, 0,
            c10::cuda::getCurrentCUDAStream()>>>(states.data_ptr<int64_t>(), t, threshold,
                0, counts.data_ptr<int64_t>(), nullptr, nullptr, nullptr, nullptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return counts;
}

template<bool WARP>
void coupled_fill_impl(at::Tensor states, std::vector<at::Tensor> ts, int n,
                       double threshold, double core, at::Tensor starts,
                       at::Tensor targets, at::Tensor values, at::Tensor diagonals) {
    check_states(states, n);
    const c10::cuda::CUDAGuard guard(states.device());
    const auto t = table_view(ts, states, n);
    check(starts, states, at::kLong);
    check(targets, states, at::kLong);
    check(values, states, at::kDouble);
    check(diagonals, states, at::kDouble);
    TORCH_CHECK(starts.numel() == states.numel()*THREADS && values.numel() == targets.numel()
                && (diagonals.numel() == 0 || diagonals.numel() == targets.numel()),
                "Invalid coupling output shapes");
    if (!states.numel()) return;
    auto stream = c10::cuda::getCurrentCUDAStream();
    if (diagonals.numel())
        coupled_kernel<true, true, WARP><<<states.numel(), THREADS, 0, stream>>>(
            states.data_ptr<int64_t>(), t, threshold, core, nullptr,
            starts.data_ptr<int64_t>(), targets.data_ptr<int64_t>(), values.data_ptr<double>(),
            diagonals.data_ptr<double>());
    else
        coupled_kernel<true, false, WARP><<<states.numel(), THREADS, 0, stream>>>(
            states.data_ptr<int64_t>(), t, threshold, core, nullptr,
            starts.data_ptr<int64_t>(), targets.data_ptr<int64_t>(), values.data_ptr<double>(), nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<bool SHARED>
at::Tensor diagonal_impl(at::Tensor states, at::Tensor h, at::Tensor j, int n, double core) {
    check_states(states, n);
    const c10::cuda::CUDAGuard guard(states.device());
    check(h, states, at::kDouble);
    check(j, states, at::kDouble);
    TORCH_CHECK(h.numel() == n && j.numel() == n*n, "Invalid diagonal table shapes");
    Tables t{};
    t.h = h.data_ptr<double>(); t.j = j.data_ptr<double>(); t.n = n;
    auto out = at::empty({states.numel()}, states.options().dtype(at::kDouble));
    if (states.numel()) {
        const int blocks = static_cast<int>(std::min<int64_t>((states.numel()+127)/128, 65535));
        if constexpr (SHARED)
            diagonal_shared_kernel<<<blocks, 128, (n+n*n)*sizeof(double), c10::cuda::getCurrentCUDAStream()>>>(
                states.data_ptr<int64_t>(), states.numel(), t, core, out.data_ptr<double>());
        else
            diagonal_kernel<<<blocks, 128, 0, c10::cuda::getCurrentCUDAStream()>>>(
                states.data_ptr<int64_t>(), states.numel(), t, core, out.data_ptr<double>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

at::Tensor coupled_counts_cuda(at::Tensor s, std::vector<at::Tensor> t, int n, double e) {
    return coupled_counts_impl<false>(s, t, n, e);
}
at::Tensor coupled_counts_warp_cuda(at::Tensor s, std::vector<at::Tensor> t, int n, double e) {
    return coupled_counts_impl<true>(s, t, n, e);
}
void coupled_fill_cuda(at::Tensor s, std::vector<at::Tensor> t, int n, double e, double c,
                      at::Tensor starts, at::Tensor targets, at::Tensor values, at::Tensor d) {
    coupled_fill_impl<false>(s, t, n, e, c, starts, targets, values, d);
}
void coupled_fill_warp_cuda(at::Tensor s, std::vector<at::Tensor> t, int n, double e, double c,
                           at::Tensor starts, at::Tensor targets, at::Tensor values, at::Tensor d) {
    coupled_fill_impl<true>(s, t, n, e, c, starts, targets, values, d);
}
at::Tensor diagonal_cuda(at::Tensor s, at::Tensor h, at::Tensor j, int n, double c) {
    return diagonal_impl<false>(s, h, j, n, c);
}
at::Tensor diagonal_shared_cuda(at::Tensor s, at::Tensor h, at::Tensor j, int n, double c) {
    return diagonal_impl<true>(s, h, j, n, c);
}
