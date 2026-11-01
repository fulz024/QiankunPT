#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/complex.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
using Bits=unsigned long long;
__device__ __forceinline__ Bits mix(Bits x) {
    x+=0x9e3779b97f4a7c15ULL;
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>27))*0x94d049bb133111ebULL;
    return x^(x>>31);
}
__device__ Bits hash_key(const int64_t* key,int words,Bits seed) {
    Bits result=seed;
    for(int w=0;w<words;++w) result=mix(result^static_cast<Bits>(key[w]));
    return result;
}
__device__ bool same_key(const int64_t* left,const int64_t* right,int words) {
    for(int w=0;w<words;++w) if(left[w]!=right[w]) return false;
    return true;
}
__global__ void insert_keys(const int64_t* keys,int64_t size,int words,int64_t* slots,Bits mask) {
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size;i+=int64_t(gridDim.x)*blockDim.x) {
        const int64_t* key=keys+i*words;
        Bits bucket=hash_key(key,words,0)&mask;
        while(true) {
            auto previous=atomicCAS(reinterpret_cast<Bits*>(slots+bucket),~Bits(0),static_cast<Bits>(i));
            if(previous==~Bits(0)||same_key(key,keys+previous*words,words)) break;
            bucket=(bucket+1)&mask;
        }
    }
}
__global__ void contains_keys(const int64_t* keys,int words,const int64_t* slots,Bits mask,
                             const int64_t* queries,int64_t count,bool* out) {
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=int64_t(gridDim.x)*blockDim.x) {
        const int64_t* key=queries+i*words;
        Bits bucket=hash_key(key,words,0)&mask; bool found=false;
        while(true) {
            int64_t slot=slots[bucket];
            if(slot<0) break;
            if(same_key(key,keys+slot*words,words)) {found=true;break;}
            bucket=(bucket+1)&mask;
        }
        out[i]=found;
    }
}
__global__ void partition_keys(const int64_t* keys,int words,int64_t count,Bits mask,Bits seed,bool* out) {
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=int64_t(gridDim.x)*blockDim.x)
        out[i]=(hash_key(keys+i*words,words,seed)&mask)==0;
}
__global__ void sketch_words(double* sketch,const int64_t* keys,int words,
        const c10::complex<double>* values,const double* denominator,const int64_t* seeds,
        int64_t count,int64_t replicas,Bits buckets) {
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=int64_t(gridDim.x)*blockDim.x) {
        double d=denominator[i],scale=sqrt(fabs(d));
        double re=values[i].real()/scale,im=values[i].imag()/scale;
        int sector=d<0;
        for(int64_t r=0;r<replicas;++r) {
            // Hash the entire key for each seed; never collapse all words to
            // one fixed fingerprint whose collisions survive every replica.
            Bits hash=hash_key(keys+i*words,words,static_cast<Bits>(seeds[r]));
            Bits bucket=hash&(buckets-1); double wr,wi;
            switch(hash>>62) {
                case 0:wr=re;wi=im;break;
                case 1:wr=-re;wi=-im;break;
                case 2:wr=-im;wi=re;break;
                default:wr=im;wi=-re;
            }
            double* dest=sketch+((r*2+sector)*buckets+bucket)*2;
            atomicAdd(dest,wr);atomicAdd(dest+1,wi);
        }
    }
}
void key_check(const at::Tensor& keys) {
    TORCH_CHECK(keys.is_cuda()&&keys.is_contiguous()&&keys.scalar_type()==at::kLong&&keys.dim()==2
        &&keys.size(1)>=1&&keys.size(1)<=8,"Expected CUDA int64 keys [rows,1..8]");
}
void vector_check(const at::Tensor& x,const at::Tensor& keys,at::ScalarType type) {
    TORCH_CHECK(x.is_cuda()&&x.device()==keys.device()&&x.is_contiguous()&&x.dim()==1&&x.scalar_type()==type,"Invalid packed-operation vector");
}
int blocks(int64_t size) {return std::min<int64_t>((size+255)/256,65535);}
}

at::Tensor packed_index_cuda(at::Tensor keys) {
    key_check(keys);const c10::cuda::CUDAGuard guard(keys.device());
    int64_t capacity=1;while(capacity<std::max<int64_t>(2,keys.size(0)*2)) capacity*=2;
    auto slots=at::full({capacity},-1,keys.options());
    if(keys.size(0)) {
        insert_keys<<<blocks(keys.size(0)),256,0,c10::cuda::getCurrentCUDAStream()>>>(keys.data_ptr<int64_t>(),keys.size(0),keys.size(1),slots.data_ptr<int64_t>(),capacity-1);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return slots;
}
at::Tensor packed_contains_cuda(at::Tensor keys,at::Tensor slots,at::Tensor queries) {
    key_check(keys);key_check(queries);vector_check(slots,keys,at::kLong);
    TORCH_CHECK(queries.device()==keys.device()&&queries.size(1)==keys.size(1)&&slots.numel()>0
        &&(slots.numel()&(slots.numel()-1))==0,"Incompatible membership index");
    const c10::cuda::CUDAGuard guard(keys.device());
    auto out=at::empty({queries.size(0)},keys.options().dtype(at::kBool));
    if(queries.size(0)) {
        contains_keys<<<blocks(queries.size(0)),256,0,c10::cuda::getCurrentCUDAStream()>>>(keys.data_ptr<int64_t>(),keys.size(1),slots.data_ptr<int64_t>(),slots.numel()-1,queries.data_ptr<int64_t>(),queries.size(0),out.data_ptr<bool>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}
void weighted_sketch_words_cuda(at::Tensor sketch,at::Tensor keys,at::Tensor values,
                               at::Tensor denominator,at::Tensor seeds,int64_t buckets) {
    key_check(keys);vector_check(values,keys,at::kComplexDouble);vector_check(denominator,keys,at::kDouble);vector_check(seeds,keys,at::kLong);
    TORCH_CHECK(buckets>0&&(buckets&(buckets-1))==0&&seeds.numel()>0,"Buckets must be a power of two; seeds nonempty");
    TORCH_CHECK(values.numel()==keys.size(0)&&denominator.numel()==keys.size(0),"Edge lengths differ");
    TORCH_CHECK(sketch.is_cuda()&&sketch.device()==keys.device()&&sketch.is_contiguous()&&sketch.scalar_type()==at::kDouble
        &&sketch.numel()==seeds.numel()*4*buckets,"Invalid multiword sketch");
    const c10::cuda::CUDAGuard guard(keys.device());
    if(keys.size(0)) {
        sketch_words<<<blocks(keys.size(0)),256,0,c10::cuda::getCurrentCUDAStream()>>>(sketch.data_ptr<double>(),keys.data_ptr<int64_t>(),keys.size(1),values.data_ptr<c10::complex<double>>(),denominator.data_ptr<double>(),seeds.data_ptr<int64_t>(),keys.size(0),seeds.numel(),buckets);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
at::Tensor packed_partition_cuda(at::Tensor keys,int64_t partitions,int64_t seed) {
    key_check(keys);const c10::cuda::CUDAGuard guard(keys.device());
    TORCH_CHECK(partitions>0&&(partitions&(partitions-1))==0,"partitions must be a positive power of two");
    auto out=at::empty({keys.size(0)},keys.options().dtype(at::kBool));
    if(keys.size(0)) {
        partition_keys<<<blocks(keys.size(0)),256,0,c10::cuda::getCurrentCUDAStream()>>>(keys.data_ptr<int64_t>(),keys.size(1),keys.size(0),partitions-1,static_cast<Bits>(seed),out.data_ptr<bool>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}
