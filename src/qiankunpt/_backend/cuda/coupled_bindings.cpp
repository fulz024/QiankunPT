#include <torch/extension.h>
#include <vector>

at::Tensor coupled_counts_cuda(at::Tensor, std::vector<at::Tensor>, int, double);
void coupled_fill_cuda(at::Tensor, std::vector<at::Tensor>, int, double, double,
                       at::Tensor, at::Tensor, at::Tensor, at::Tensor);
at::Tensor diagonal_cuda(at::Tensor, at::Tensor, at::Tensor, int, double);
at::Tensor diagonal_shared_cuda(at::Tensor, at::Tensor, at::Tensor, int, double);
at::Tensor coupled_counts_warp_cuda(at::Tensor, std::vector<at::Tensor>, int, double);
void coupled_fill_warp_cuda(at::Tensor, std::vector<at::Tensor>, int, double, double,
                            at::Tensor, at::Tensor, at::Tensor, at::Tensor);
void weighted_sketch_cuda(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t);
void weighted_sketch_replica_cuda(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t);
at::Tensor coupled_counts_words_cuda(at::Tensor, std::vector<at::Tensor>, int, double, at::Tensor, bool);
void coupled_fill_words_cuda(at::Tensor, std::vector<at::Tensor>, int, double, at::Tensor, double, bool,
                            at::Tensor, at::Tensor, at::Tensor, at::Tensor);
at::Tensor diagonal_words_cuda(at::Tensor, at::Tensor, at::Tensor, int, double, bool);
at::Tensor packed_index_cuda(at::Tensor);
at::Tensor packed_contains_cuda(at::Tensor, at::Tensor, at::Tensor);
void weighted_sketch_words_cuda(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t);
std::vector<at::Tensor> stream_external_words_cuda(at::Tensor,std::vector<at::Tensor>,int,double,
    at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,double,at::Tensor,at::Tensor,int64_t,bool,int,bool,int64_t,int64_t);
std::vector<at::Tensor> stream_external_pair_cuda(at::Tensor,std::vector<at::Tensor>,int,double,
    at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,double,at::Tensor,at::Tensor,int64_t,bool,int,bool,int64_t,int64_t);
at::Tensor packed_partition_cuda(at::Tensor,int64_t,int64_t);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("counts", &coupled_counts_cuda);
    m.def("fill", &coupled_fill_cuda);
    m.def("diagonal", &diagonal_cuda);
    m.def("diagonal_shared", &diagonal_shared_cuda);
    m.def("counts_warp", &coupled_counts_warp_cuda);
    m.def("fill_warp", &coupled_fill_warp_cuda);
    m.def("weighted_sketch", &weighted_sketch_cuda);
    m.def("weighted_sketch_replica", &weighted_sketch_replica_cuda);
    m.def("counts_words", &coupled_counts_words_cuda);
    m.def("fill_words", &coupled_fill_words_cuda);
    m.def("diagonal_words", &diagonal_words_cuda);
    m.def("packed_index", &packed_index_cuda);
    m.def("packed_contains", &packed_contains_cuda);
    m.def("weighted_sketch_words", &weighted_sketch_words_cuda);
    m.def("stream_external_words", &stream_external_words_cuda);
    m.def("stream_external_pair", &stream_external_pair_cuda);
    m.def("packed_partition", &packed_partition_cuda);
}
