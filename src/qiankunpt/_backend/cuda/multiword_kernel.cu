#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/complex.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include "screening.cuh"

namespace {
using Bits = unsigned long long;
constexpr int THREADS = 128, MAX_ORBITALS = 512;
struct Tables {
    const int64_t *so, *d_o;
    const int32_t *sa, *da;
    const double *sv, *dv, *h, *j;
    int n;
};

template<int W> struct Key {
    Bits words[W];
    __device__ explicit Key(const int64_t* p) {
        #pragma unroll
        for (int w=0; w<W; ++w) words[w] = static_cast<Bits>(p[w]);
    }
    __device__ bool occupied(int i) const { return (words[i/64] >> (i%64)) & 1; }
    __device__ void flip(int i) { words[i/64] ^= Bits(1) << (i%64); }
    __device__ void store(int64_t* p) const {
        #pragma unroll
        for (int w=0; w<W; ++w) p[w] = static_cast<int64_t>(words[w]);
    }
    // Same half-open interval convention as the validated single-word kernel.
    __device__ int parity(int i, int a) const {
        int lo=min(i,a), hi=max(i,a), result=0;
        #pragma unroll
        for (int w=0; w<W; ++w) {
            int l=max(0,lo-64*w), h=min(64,hi-64*w);
            if (l<h) {
                Bits upper = h==64 ? ~Bits(0) : (Bits(1)<<h)-1;
                Bits lower = (Bits(1)<<l)-1;
                result ^= __popcll(words[w] & (upper ^ lower)) & 1;
            }
        }
        return result;
    }
    __device__ int decode(int* occupied) const {
        int count=0;
        #pragma unroll
        for (int w=0; w<W; ++w) {
            Bits rest=words[w];
            while(rest) { occupied[count++]=64*w+__ffsll(rest)-1; rest&=rest-1; }
        }
        return count;
    }
};

// Canonical diagonal sum: h_i, then all J_ik with k>i, then core energy.
__device__ double diagonal_indices(const int* occ, int count, Tables t, double core) {
    double value=0;
    for(int p=0;p<count;++p) {
        int i=occ[p]; value+=t.h[i];
        for(int q=p+1;q<count;++q) value+=t.j[i*t.n+occ[q]];
    }
    return value+core;
}

template<int W> __device__ double diagonal_bits(const Key<W>& key, Tables t, double core) {
    double value=0;
    for(int w=0;w<W;++w) {
        Bits state=key.words[w];
        while(state) {
            int i=w*64+__ffsll(state)-1; state&=state-1; value+=t.h[i];
            for(int v=w;v<W;++v) {
                Bits rest=v==w ? state : key.words[v];
                while(rest) { int k=v*64+__ffsll(rest)-1; rest&=rest-1; value+=t.j[i*t.n+k]; }
            }
        }
    }
    return value+core;
}

template<int W, bool WRITE, bool DIAG>
__global__ void expand(const int64_t* states, Tables t, double threshold,
        const double* thresholds, double core, bool self, int64_t* counts,
        const int64_t* starts, int64_t* targets, double* values, double* diagonals) {
    __shared__ int occupied[MAX_ORBITALS], nocc;
    __shared__ double fields[MAX_ORBITALS], source_diag;
    const int row=blockIdx.x, lane=threadIdx.x, group=lane/32, member=lane%32;
    const Key<W> source(states+int64_t(row)*W);
    const double eps=thresholds ? thresholds[row] : threshold;
    if(lane==0) {
        nocc=source.decode(occupied);
        if constexpr(WRITE) {
            if(self || DIAG) source_diag=diagonal_indices(occupied,nocc,t,core);
        }
    }
    __syncthreads();
    if constexpr(WRITE && DIAG) {
        for(int a=lane;a<t.n;a+=THREADS) {
            double f=t.h[a];
            for(int p=0;p<nocc;++p) f+=t.j[a*t.n+occupied[p]];
            fields[a]=f;
        }
        __syncthreads();
    }
    int64_t cursor=WRITE ? starts[int64_t(row)*THREADS+lane] : 0;
    const int64_t initial=cursor;
    if(self && lane==0) {
        if constexpr(WRITE) {
            source.store(targets+cursor*W); values[cursor]=source_diag;
            if constexpr(DIAG) diagonals[cursor]=source_diag;
        }
        ++cursor;
    }
    for(int oi=0;oi<nocc;++oi) {
        int i=occupied[oi];
        for(int64_t e=oi%4==group ? t.so[i]+member : t.so[i+1];e<t.so[i+1];e+=32) {
            int a=t.sa[e]; if(source.occupied(a)) continue;
            const double* integrals=t.sv+e*(t.n+1);
            double v=integrals[0];
            for(int p=0;p<nocc;++p) v+=integrals[1+occupied[p]];
            if(!keep_pt_coupling(v,eps)) continue;
            if constexpr(WRITE) {
                Key<W> target=source; target.flip(i);
                if(target.parity(i,a)) v=-v;
                target.flip(a); target.store(targets+cursor*W); values[cursor]=v;
                if constexpr(DIAG) diagonals[cursor]=source_diag-fields[i]+fields[a]-t.j[i*t.n+a];
            }
            ++cursor;
        }
        for(int oj=oi+1+group;oj<nocc;oj+=4) {
            int j=occupied[oj], ij=i*t.n+j;
            for(int64_t e=t.d_o[ij]+member;e<t.d_o[ij+1];e+=32) {
                // The exported double lists are sorted by decreasing |H|.
                double v=t.dv[e]; if(!keep_pt_coupling(v,eps)) break;
                int a=t.da[2*e], b=t.da[2*e+1];
                if(source.occupied(a)||source.occupied(b)) continue;
                if constexpr(WRITE) {
                    Key<W> target=source; target.flip(i);
                    int sign=target.parity(i,a); target.flip(a); target.flip(j);
                    sign^=target.parity(j,b); target.flip(b);
                    if(sign) v=-v;
                    target.store(targets+cursor*W); values[cursor]=v;
                    if constexpr(DIAG) diagonals[cursor]=source_diag-fields[i]-fields[j]+fields[a]+fields[b]
                        +t.j[i*t.n+j]+t.j[a*t.n+b]-t.j[i*t.n+a]-t.j[i*t.n+b]
                        -t.j[j*t.n+a]-t.j[j*t.n+b];
                }
                ++cursor;
            }
        }
    }
    if constexpr(!WRITE) counts[int64_t(row)*THREADS+lane]=cursor-initial;
}

__device__ __forceinline__ Bits mix_key(Bits x) {
    x+=0x9e3779b97f4a7c15ULL;
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>27))*0x94d049bb133111ebULL;
    return x^(x>>31);
}
template<int W> __device__ Bits hash_words(const Key<W>& key, Bits seed) {
    #pragma unroll
    for(int w=0;w<W;++w) seed=mix_key(seed^key.words[w]);
    return seed;
}

// Consume one external edge immediately. No external determinant buffer or
// count/fill pass is needed. Membership always compares the complete key.
template<int W, bool REAL, bool PAIRED=false> struct StreamConsumer {
    const int64_t *support, *slots, *seeds;
    Bits mask, buckets;
    Bits partition_mask, partition_seed;
    int replicas;
    Tables table;
    double core, energy, epsilon, re, im, inverse_count;
    const double* fields;
    double source_diagonal;
    bool incremental;
    double* sketch;
    double* coarse_sketch;
    double coarse_threshold;
    double self_sum=0;
    double coarse_self_sum=0;
    int64_t edges=0;
    int64_t coarse_edges=0;
    __device__ __forceinline__ void add(const Key<W>& source, const Key<W>& target,
            double h, int i, int j, int a, int b, bool* ordered_coarse_active=nullptr) {
        if(partition_mask && (hash_words(target,partition_seed)&partition_mask))return;
        Bits bucket=hash_words(target,0)&mask;
        while(true) {
            int64_t slot=slots[bucket];
            if(slot<0) break;
            bool same=true;
            #pragma unroll
            for(int w=0;w<W;++w) same &= target.words[w]==static_cast<Bits>(support[slot*W+w]);
            if(same) return;
            bucket=(bucket+1)&mask;
        }
        // Only retained external targets need a fermionic phase or diagonal.
        Key<W> phase_key=source;phase_key.flip(i);
        int sign=phase_key.parity(i,a);
        if(j>=0) {phase_key.flip(a);phase_key.flip(j);sign^=phase_key.parity(j,b);}
        if(sign)h=-h;
        double proposed_diagonal=0;
        if(incremental) {
            proposed_diagonal=j<0?source_diagonal-fields[i]+fields[a]-table.j[i*table.n+a]
                :source_diagonal-fields[i]-fields[j]+fields[a]+fields[b]
                  +table.j[i*table.n+j]+table.j[a*table.n+b]-table.j[i*table.n+a]
                  -table.j[i*table.n+b]-table.j[j*table.n+a]-table.j[j*table.n+b];
        }
        // Canonical target-only denominator: independent of which source
        // generated this edge, including near-zero regularized denominators.
        double diag=proposed_diagonal;
        // The incremental expression is algebraically identical but its last
        // bits depend on the source. Near cancellation, always use canonical
        // target evaluation before denominator regularization.
        if(!incremental || fabs(energy-diag)<1e-4) {
            int count=0;
            #pragma unroll
            for(int w=0;w<W;++w) count+=__popcll(target.words[w]);
            if(count<=32) { int occ[32];target.decode(occ);diag=diagonal_indices(occ,count,table,core); }
            else diag=diagonal_bits(target,table,core);
        }
        double d=energy-diag;
        if(fabs(d)<epsilon) d=d>=0?epsilon:-epsilon;
        double vr=h*re,vi=REAL?0:h*im;
        double self_value=(vr*vr+vi*vi)*inverse_count/d;
        self_sum+=self_value;
        ++edges;
        bool coarse=false;
        if constexpr(PAIRED) {
            // Retained doubles form a monotone subsequence of their ordered
            // occupied-pair/lane list. Once below coarse, stay fine-only.
            // Singles pass nullptr: their occupied-dependent H is not ordered.
            if(ordered_coarse_active && !*ordered_coarse_active) coarse=false;
            else {
                coarse=keep_pt_coupling(h,coarse_threshold);
                if(ordered_coarse_active) *ordered_coarse_active=coarse;
            }
            if(coarse) {coarse_self_sum+=self_value;++coarse_edges;}
        }
        double scale=sqrt(fabs(d));vr/=scale;vi/=scale;
        int sector=d<0;
        for(int r=0;r<replicas;++r) {
            Bits hash=hash_words(target,static_cast<Bits>(seeds[r]));
            int64_t offset=((int64_t(r)*2+sector)*buckets+(hash&(buckets-1)))*2;
            double* dest=sketch+offset;
            if constexpr(REAL) {
                // A real coefficient under a fourth-root-of-unity sign has
                // exactly one nonzero component: one FP64 atomic per replica.
                switch(hash>>62) {
                    case 0:atomicAdd(dest,vr);break;
                    case 1:atomicAdd(dest,-vr);break;
                    case 2:atomicAdd(dest+1,vr);break;
                    default:atomicAdd(dest+1,-vr);
                }
                if constexpr(PAIRED) {
                    if(coarse) {
                        double* other=coarse_sketch+offset;
                        switch(hash>>62) {
                            case 0:atomicAdd(other,vr);break;
                            case 1:atomicAdd(other,-vr);break;
                            case 2:atomicAdd(other+1,vr);break;
                            default:atomicAdd(other+1,-vr);
                        }
                    }
                }
            } else {
                double wr,wi;
                switch(hash>>62) {
                    case 0:wr=vr;wi=vi;break;
                    case 1:wr=-vr;wi=-vi;break;
                    case 2:wr=-vi;wi=vr;break;
                    default:wr=vi;wi=-vr;
                }
                atomicAdd(dest,wr);atomicAdd(dest+1,wi);
                if constexpr(PAIRED) {
                    if(coarse) {atomicAdd(coarse_sketch+offset,wr);atomicAdd(coarse_sketch+offset+1,wi);}
                }
            }
        }
    }
};

template<int W, bool REAL, bool PAIRED=false>
__global__ void stream_external_energy(const int64_t* states, Tables t, const double* thresholds,
        const c10::complex<double>* weights,const double* self_factors,
        const double* energy, StreamConsumer<W,REAL,PAIRED> consumer, double* self_out,int64_t* edges_out,
        const double* coarse_thresholds) {
    __shared__ int occupied[MAX_ORBITALS],nocc;
    __shared__ double fields[MAX_ORBITALS],source_diagonal;
    int row=blockIdx.x,lane=threadIdx.x,group=4*blockIdx.y+lane/32,member=lane%32;
    const int groups=4*gridDim.y;
    Key<W> source(states+int64_t(row)*W);
    if(lane==0) nocc=source.decode(occupied);
    __syncthreads();
    if(consumer.incremental) {
        if(lane==0)source_diagonal=diagonal_indices(occupied,nocc,t,consumer.core);
        for(int a=lane;a<t.n;a+=THREADS) {
            double f=t.h[a];
            for(int p=0;p<nocc;++p)f+=t.j[a*t.n+occupied[p]];
            fields[a]=f;
        }
        __syncthreads();
    }
    consumer.energy=*energy;consumer.re=weights[row].real();consumer.im=weights[row].imag();
    consumer.inverse_count=self_factors[row];
    consumer.fields=fields;consumer.source_diagonal=consumer.incremental?source_diagonal:0;
    double eps=thresholds[row];
    if constexpr(PAIRED) consumer.coarse_threshold=coarse_thresholds[row];
    for(int oi=0;oi<nocc;++oi) {
        int i=occupied[oi];
        for(int64_t e=oi%groups==group?t.so[i]+member:t.so[i+1];e<t.so[i+1];e+=32) {
            int a=t.sa[e];if(source.occupied(a))continue;
            const double* integrals=t.sv+e*(t.n+1);double v=integrals[0];
            for(int p=0;p<nocc;++p)v+=integrals[1+occupied[p]];
            if(!keep_pt_coupling(v,eps))continue;
            Key<W> target=source;target.flip(i);target.flip(a);
            consumer.add(source,target,v,i,-1,a,-1);
        }
        int pair_start=oi*(2*nocc-oi-1)/2;
        int offset=(group-pair_start%groups+groups)%groups;
        for(int oj=oi+1+offset;oj<nocc;oj+=groups) {
            int j=occupied[oj],ij=i*t.n+j;
            // Thread-local state resets for every occupied-pair list/lane.
            bool coarse_active=true;
            for(int64_t e=t.d_o[ij]+member;e<t.d_o[ij+1];e+=32) {
                double v=t.dv[e];if(!keep_pt_coupling(v,eps))break;
                int a=t.da[2*e],b=t.da[2*e+1];
                if(source.occupied(a)||source.occupied(b))continue;
                Key<W> target=source;target.flip(i);target.flip(a);target.flip(j);target.flip(b);
                consumer.add(source,target,v,i,j,a,b,&coarse_active);
            }
        }
    }
    int64_t out=(int64_t(row)*gridDim.y+blockIdx.y)*THREADS+lane;
    self_out[out]=consumer.self_sum;
    edges_out[out]=consumer.edges;
    if constexpr(PAIRED) {
        int64_t stride=int64_t(gridDim.x)*gridDim.y*THREADS;
        self_out[stride+out]=consumer.coarse_self_sum;
        edges_out[stride+out]=consumer.coarse_edges;
    }
}

template<int W, bool INDEXED>
__global__ void diagonal(const int64_t* states,int64_t size,Tables t,double core,double* out) {
    for(int64_t row=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;row<size;row+=int64_t(gridDim.x)*blockDim.x) {
        Key<W> key(states+row*W);
        if constexpr(INDEXED) {
            int count=0;
            #pragma unroll
            for(int w=0;w<W;++w) count+=__popcll(key.words[w]);
            if(count<=32) {
                int occ[32]; key.decode(occ);
                out[row]=diagonal_indices(occ,count,t,core);
            } else out[row]=diagonal_bits(key,t,core);
        } else out[row]=diagonal_bits(key,t,core);
    }
}

void check_vector(const at::Tensor& x,const at::Tensor& states,at::ScalarType type) {
    TORCH_CHECK(x.is_cuda() && x.device()==states.device() && x.is_contiguous()
        && x.dim()==1 && x.scalar_type()==type,"Invalid multiword vector");
}
int check_states(const at::Tensor& states,int n) {
    int words=(n+63)/64;
    TORCH_CHECK(n>0 && n<=MAX_ORBITALS && states.is_cuda() && states.is_contiguous()
        && states.scalar_type()==at::kLong && states.dim()==2 && states.size(1)==words,
        "Multiword states require contiguous CUDA int64 [rows, ceil(n/64)], n<=512");
    TORCH_CHECK(states.size(0)<=2147483647,"Too many source rows");
    return words;
}
Tables tables(const std::vector<at::Tensor>& ts,const at::Tensor& states,int n) {
    TORCH_CHECK(ts.size()==8,"Expected eight Hamiltonian tables");
    const at::ScalarType types[]={at::kLong,at::kInt,at::kDouble,at::kLong,at::kInt,at::kDouble,at::kDouble,at::kDouble};
    for(int k=0;k<8;++k) check_vector(ts[k],states,types[k]);
    TORCH_CHECK(ts[0].numel()==n+1 && ts[3].numel()==n*n+1 && ts[2].numel()==ts[1].numel()*(n+1)
        && ts[4].numel()==ts[5].numel()*2 && ts[6].numel()==n && ts[7].numel()==n*n,"Invalid multiword tables");
    return {ts[0].data_ptr<int64_t>(),ts[3].data_ptr<int64_t>(),ts[1].data_ptr<int32_t>(),ts[4].data_ptr<int32_t>(),
        ts[2].data_ptr<double>(),ts[5].data_ptr<double>(),ts[6].data_ptr<double>(),ts[7].data_ptr<double>(),n};
}

#define DISPATCH_WORDS(CALL) switch(words) { \
    case 1: { constexpr int W=1; CALL; break; } case 2: { constexpr int W=2; CALL; break; } \
    case 3: { constexpr int W=3; CALL; break; } case 4: { constexpr int W=4; CALL; break; } \
    case 5: { constexpr int W=5; CALL; break; } case 6: { constexpr int W=6; CALL; break; } \
    case 7: { constexpr int W=7; CALL; break; } case 8: { constexpr int W=8; CALL; break; } }
}

at::Tensor coupled_counts_words_cuda(at::Tensor states,std::vector<at::Tensor> ts,int n,
        double threshold,at::Tensor thresholds,bool self) {
    int words=check_states(states,n); const c10::cuda::CUDAGuard guard(states.device());
    auto t=tables(ts,states,n); check_vector(thresholds,states,at::kDouble);
    TORCH_CHECK(std::isfinite(threshold)&&threshold>=0 && (thresholds.numel()==0||thresholds.numel()==states.size(0)),"Invalid thresholds");
    auto out=at::empty({states.size(0)*THREADS},states.options());
    if(states.size(0)) {
        auto stream=c10::cuda::getCurrentCUDAStream();
        const double* ep=thresholds.numel() ? thresholds.data_ptr<double>() : nullptr;
        DISPATCH_WORDS((expand<W,false,false><<<states.size(0),THREADS,0,stream>>>(states.data_ptr<int64_t>(),t,threshold,ep,0,self,out.data_ptr<int64_t>(),nullptr,nullptr,nullptr,nullptr)));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

void coupled_fill_words_cuda(at::Tensor states,std::vector<at::Tensor> ts,int n,double threshold,
        at::Tensor thresholds,double core,bool self,at::Tensor starts,at::Tensor targets,at::Tensor values,at::Tensor diag) {
    int words=check_states(states,n); const c10::cuda::CUDAGuard guard(states.device());
    auto t=tables(ts,states,n); check_vector(thresholds,states,at::kDouble);
    check_vector(starts,states,at::kLong); check_vector(values,states,at::kDouble); check_vector(diag,states,at::kDouble);
    TORCH_CHECK(thresholds.numel()==0||thresholds.numel()==states.size(0),"Invalid thresholds");
    TORCH_CHECK(targets.is_cuda()&&targets.device()==states.device()&&targets.is_contiguous()&&targets.scalar_type()==at::kLong
        &&targets.dim()==2&&targets.size(1)==words&&targets.size(0)==values.numel()&&starts.numel()==states.size(0)*THREADS
        &&(diag.numel()==0||diag.numel()==values.numel()),"Invalid multiword outputs");
    if(!states.size(0)) return;
    auto stream=c10::cuda::getCurrentCUDAStream(); const double* ep=thresholds.numel()?thresholds.data_ptr<double>():nullptr;
    if(diag.numel()) {
        DISPATCH_WORDS((expand<W,true,true><<<states.size(0),THREADS,0,stream>>>(states.data_ptr<int64_t>(),t,threshold,ep,core,self,nullptr,starts.data_ptr<int64_t>(),targets.data_ptr<int64_t>(),values.data_ptr<double>(),diag.data_ptr<double>())));
    } else {
        DISPATCH_WORDS((expand<W,true,false><<<states.size(0),THREADS,0,stream>>>(states.data_ptr<int64_t>(),t,threshold,ep,core,self,nullptr,starts.data_ptr<int64_t>(),targets.data_ptr<int64_t>(),values.data_ptr<double>(),nullptr)));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor diagonal_words_cuda(at::Tensor states,at::Tensor h,at::Tensor j,int n,double core,bool indexed) {
    int words=check_states(states,n); const c10::cuda::CUDAGuard guard(states.device());
    check_vector(h,states,at::kDouble); check_vector(j,states,at::kDouble);
    TORCH_CHECK(h.numel()==n&&j.numel()==n*n,"Invalid diagonal tables");
    Tables t{};t.h=h.data_ptr<double>();t.j=j.data_ptr<double>();t.n=n;
    auto out=at::empty({states.size(0)},states.options().dtype(at::kDouble));
    if(states.size(0)) {
        int blocks=std::min<int64_t>((states.size(0)+127)/128,65535); auto stream=c10::cuda::getCurrentCUDAStream();
        if(indexed) { DISPATCH_WORDS((diagonal<W,true><<<blocks,128,0,stream>>>(states.data_ptr<int64_t>(),states.size(0),t,core,out.data_ptr<double>()))); }
        else { DISPATCH_WORDS((diagonal<W,false><<<blocks,128,0,stream>>>(states.data_ptr<int64_t>(),states.size(0),t,core,out.data_ptr<double>()))); }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

static std::vector<at::Tensor> stream_external_words_impl(at::Tensor states,std::vector<at::Tensor> ts,int n,
        double core,at::Tensor support,at::Tensor slots,at::Tensor thresholds,at::Tensor weights,
        at::Tensor counts,at::Tensor energy,double epsilon,at::Tensor sketch,at::Tensor seeds,
        int64_t buckets,bool real,int shards,bool incremental,int64_t partitions,int64_t partition_seed,
        at::Tensor coarse_thresholds) {
    int words=check_states(states,n);check_states(support,n);
    const c10::cuda::CUDAGuard guard(states.device());auto t=tables(ts,states,n);
    check_vector(slots,states,at::kLong);check_vector(thresholds,states,at::kDouble);
    check_vector(weights,states,at::kComplexDouble);check_vector(counts,states,at::kDouble);
    check_vector(seeds,states,at::kLong);
    TORCH_CHECK(support.device()==states.device()&&slots.numel()>0&&(slots.numel()&(slots.numel()-1))==0,"Invalid support index");
    TORCH_CHECK(weights.numel()==states.size(0)&&counts.numel()==states.size(0)&&thresholds.numel()==states.size(0),"Source lengths differ");
    TORCH_CHECK(buckets>0&&(buckets&(buckets-1))==0&&seeds.numel()>0&&seeds.numel()<=2147483647,"Invalid sketch size");
    bool paired=coarse_thresholds.defined();
    if(paired) {
        check_vector(coarse_thresholds,states,at::kDouble);
        TORCH_CHECK(coarse_thresholds.numel()==states.size(0),"Coarse source lengths differ");
    }
    TORCH_CHECK(sketch.is_cuda()&&sketch.device()==states.device()&&sketch.is_contiguous()&&sketch.scalar_type()==at::kDouble
        &&sketch.numel()==(paired?2:1)*seeds.numel()*4*buckets,"Invalid sketch buffer");
    TORCH_CHECK(energy.is_cuda()&&energy.device()==states.device()&&energy.scalar_type()==at::kDouble&&energy.numel()==1,"Invalid reference energy");
    TORCH_CHECK(std::isfinite(epsilon)&&epsilon>0,"Invalid denominator floor");
    TORCH_CHECK(shards>0&&shards<=32,"stream shards must be 1..32");
    TORCH_CHECK(partitions>0&&(partitions&(partitions-1))==0,"external partitions must be a power of two");
    int64_t parts=states.size(0)*shards*THREADS;
    auto shape=paired?std::vector<int64_t>{2,parts}:std::vector<int64_t>{parts};
    auto self=at::empty(shape,states.options().dtype(at::kDouble));
    auto edges=at::empty(shape,states.options());
    if(!states.size(0))return {self,edges};
    auto stream=c10::cuda::getCurrentCUDAStream();
    // The scalar energy stays on the GPU. Loaded in the launch wrapper
    // kernel below, never via Tensor::item() on the host.
    #define STREAM_CALL(REAL,PAIRED) \
        StreamConsumer<W,REAL,PAIRED> consumer{}; \
        consumer.support=support.data_ptr<int64_t>();consumer.slots=slots.data_ptr<int64_t>();consumer.seeds=seeds.data_ptr<int64_t>(); \
        consumer.mask=slots.numel()-1;consumer.buckets=buckets;consumer.replicas=seeds.numel();consumer.table=t; \
        consumer.core=core;consumer.epsilon=epsilon;consumer.sketch=sketch.data_ptr<double>();consumer.incremental=incremental; \
        consumer.coarse_sketch=paired?sketch.data_ptr<double>()+seeds.numel()*4*buckets:nullptr; \
        consumer.partition_mask=partitions-1;consumer.partition_seed=static_cast<Bits>(partition_seed); \
        stream_external_energy<W,REAL,PAIRED><<<dim3(states.size(0),shards),THREADS,0,stream>>>(states.data_ptr<int64_t>(),t,thresholds.data_ptr<double>(),weights.data_ptr<c10::complex<double>>(),counts.data_ptr<double>(),energy.data_ptr<double>(),consumer,self.data_ptr<double>(),edges.data_ptr<int64_t>(),paired?coarse_thresholds.data_ptr<double>():nullptr)
    if(paired) {
        if(real) { DISPATCH_WORDS(STREAM_CALL(true,true)); }
        else { DISPATCH_WORDS(STREAM_CALL(false,true)); }
    } else {
        if(real) { DISPATCH_WORDS(STREAM_CALL(true,false)); }
        else { DISPATCH_WORDS(STREAM_CALL(false,false)); }
    }
    #undef STREAM_CALL
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {self,edges};
}

std::vector<at::Tensor> stream_external_words_cuda(at::Tensor states,std::vector<at::Tensor> ts,int n,
        double core,at::Tensor support,at::Tensor slots,at::Tensor thresholds,at::Tensor weights,
        at::Tensor counts,at::Tensor energy,double epsilon,at::Tensor sketch,at::Tensor seeds,
        int64_t buckets,bool real,int shards,bool incremental,int64_t partitions,int64_t partition_seed) {
    return stream_external_words_impl(states,ts,n,core,support,slots,thresholds,weights,counts,energy,
        epsilon,sketch,seeds,buckets,real,shards,incremental,partitions,partition_seed,at::Tensor());
}

std::vector<at::Tensor> stream_external_pair_cuda(at::Tensor states,std::vector<at::Tensor> ts,int n,
        double core,at::Tensor support,at::Tensor slots,at::Tensor fine,at::Tensor coarse,at::Tensor weights,
        at::Tensor counts,at::Tensor energy,double epsilon,at::Tensor sketch,at::Tensor seeds,
        int64_t buckets,bool real,int shards,bool incremental,int64_t partitions,int64_t partition_seed) {
    return stream_external_words_impl(states,ts,n,core,support,slots,fine,weights,counts,energy,
        epsilon,sketch,seeds,buckets,real,shards,incremental,partitions,partition_seed,coarse);
}
