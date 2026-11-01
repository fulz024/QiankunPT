// Real spatial-orbital FCIDUMP to the CUDA excitation-table format.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
namespace py = pybind11;

struct Integral { int p, q, r, s; double value; };
using Integrals = std::unordered_map<std::uint64_t, Integral>;
struct DoubleEntry { int a, b; double value; };

std::uint64_t pair_key(std::uint64_t p, std::uint64_t q) {
    if (p < q) std::swap(p, q);
    return p * (p + 1) / 2 + q;
}

std::uint64_t integral_key(int p, int q, int r, int s) {
    return pair_key(pair_key(p, q), pair_key(r, s));
}

// <ij|ab> = (i a | j b), with a Kronecker delta on each electron's spin.
double two_body(const Integrals& eri, int i, int j, int a, int b) {
    if ((i & 1) != (a & 1) || (j & 1) != (b & 1)) return 0;
    auto entry = eri.find(integral_key(i / 2, a / 2, j / 2, b / 2));
    return entry == eri.end() ? 0 : entry->second.value;
}

double antisymmetrized(const Integrals& eri, int i, int j, int a, int b) {
    return two_body(eri, i, j, a, b) - two_body(eri, i, j, b, a);
}

template<class T>
py::array_t<T> array(const std::vector<T>& values) {
    py::array_t<T> result(values.size());
    if (!values.empty())
        std::memcpy(result.mutable_data(), values.data(), values.size() * sizeof(T));
    return result;
}

py::dict load_tables(const std::string& path, int spatial_orbitals, int electrons, int ms2) {
    if (spatial_orbitals < 1 || spatial_orbitals > 256 || ms2 < 0 ||
        electrons < ms2 || electrons > 2 * spatial_orbitals || (electrons - ms2) % 2)
        throw std::invalid_argument("Invalid orbital count or electron sector");
    const int n = 2 * spatial_orbitals;
    std::ifstream input(path);
    if (!input) throw std::invalid_argument("Cannot open FCIDUMP");
    std::string line;
    bool header_end = false;
    while (std::getline(input, line)) {
        if (line.find("&END") != std::string::npos || line.find('/') != std::string::npos) {
            header_end = true;
            break;
        }
    }
    if (!header_end) throw std::invalid_argument("Missing FCIDUMP header terminator");

    // Store one value per eightfold-symmetric spatial integral.
    std::vector<double> one(spatial_orbitals * spatial_orbitals, 0);
    Integrals eri;
    double core = 0;
    bool found_core = false;
    while (std::getline(input, line)) {
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
        std::istringstream record(line);
        double value;
        int p, q, r, s;
        std::string extra;
        if (!(record >> value >> p >> q >> r >> s) || (record >> extra) ||
            !std::isfinite(value) || std::min({p,q,r,s}) < 0 ||
            std::max({p,q,r,s}) > spatial_orbitals)
            throw std::invalid_argument("Invalid FCIDUMP integral record");
        if (p && q && r && s) {
            --p; --q; --r; --s;
            eri.emplace(integral_key(p,q,r,s), Integral{p,q,r,s,value});
        } else if (p && q && !r && !s) {
            --p; --q;
            one[p * spatial_orbitals + q] = one[q * spatial_orbitals + p] = value;
        } else if (p && !q && !r && !s) {
            // Optional orbital-energy records are separate from Hamiltonian integrals.
        } else if (!p && !q && !r && !s) {
            core = value;
            found_core = true;
        } else {
            throw std::invalid_argument("Malformed FCIDUMP integral indices");
        }
    }
    if (!found_core) throw std::invalid_argument("Missing FCIDUMP core-energy record");

    // Single i -> a: h_ia + sum over occupied k of <ik||ak>.
    std::vector<std::int64_t> single_offsets{0};
    std::vector<std::int32_t> single_targets;
    std::vector<double> single_values;
    struct SingleEntry { int a; std::vector<double> values; };
    for (int i = 0; i < n; ++i) {
        std::vector<SingleEntry> row;
        for (int a = i % 2; a < n; a += 2) {
            std::vector<double> values(n + 1);
            values[0] = one[(i / 2) * spatial_orbitals + a / 2];
            bool nonzero = values[0] != 0;
            for (int k = 0; k < n; ++k) {
                values[k + 1] = antisymmetrized(eri, i, k, a, k);
                nonzero = nonzero || values[k + 1] != 0;
            }
            if (nonzero) row.push_back({a, std::move(values)});
        }
        std::sort(row.begin(), row.end(), [](const SingleEntry& x, const SingleEntry& y) {
            const double a = std::abs(x.values[0]), b = std::abs(y.values[0]);
            return a != b ? a > b : x.a < y.a;
        });
        for (const auto& entry : row) {
            single_targets.push_back(entry.a);
            single_values.insert(single_values.end(), entry.values.begin(), entry.values.end());
        }
        single_offsets.push_back(single_targets.size());
    }

    // Generate double candidates from the sparse spatial integrals, not an n^4 scan.
    std::vector<std::vector<DoubleEntry>> rows(n * n);
    for (const auto& item : eri) {
        const auto& g = item.second;
        const std::array<std::array<int,4>,8> permutations{{
            {{g.p,g.q,g.r,g.s}}, {{g.q,g.p,g.r,g.s}},
            {{g.p,g.q,g.s,g.r}}, {{g.q,g.p,g.s,g.r}},
            {{g.r,g.s,g.p,g.q}}, {{g.s,g.r,g.p,g.q}},
            {{g.r,g.s,g.q,g.p}}, {{g.s,g.r,g.q,g.p}}
        }};
        for (const auto& pqrs : permutations) {
            for (int spin_i = 0; spin_i < 2; ++spin_i) {
                for (int spin_j = 0; spin_j < 2; ++spin_j) {
                    int i = 2 * pqrs[0] + spin_i, j = 2 * pqrs[2] + spin_j;
                    int a = 2 * pqrs[1] + spin_i, b = 2 * pqrs[3] + spin_j;
                    if (i == j || a == b) continue;
                    if (i > j) { std::swap(i,j); std::swap(a,b); }
                    // Equal-spin target pairs have one antisymmetrized orientation.
                    if ((i & 1) == (j & 1) && a > b) std::swap(a,b);
                    const double value = antisymmetrized(eri, i, j, a, b);
                    if (value != 0) rows[i * n + j].push_back({a,b,value});
                }
            }
        }
    }
    std::vector<std::int64_t> double_offsets{0};
    std::vector<std::int32_t> double_targets;
    std::vector<double> double_values;
    for (auto& row : rows) {
        std::sort(row.begin(), row.end(), [](const DoubleEntry& x, const DoubleEntry& y) {
            return std::make_pair(x.a,x.b) < std::make_pair(y.a,y.b);
        });
        row.erase(std::unique(row.begin(), row.end(), [](const DoubleEntry& x, const DoubleEntry& y) {
            return x.a == y.a && x.b == y.b;
        }), row.end());
        std::sort(row.begin(), row.end(), [](const DoubleEntry& x, const DoubleEntry& y) {
            const double a = std::abs(x.value), b = std::abs(y.value);
            return a != b ? a > b : std::make_pair(x.a,x.b) < std::make_pair(y.a,y.b);
        });
        for (const auto& entry : row) {
            double_targets.push_back(entry.a);
            double_targets.push_back(entry.b);
            double_values.push_back(entry.value);
        }
        double_offsets.push_back(double_values.size());
    }

    std::vector<double> diagonal_one(n), diagonal_two(n * n, 0);
    for (int i = 0; i < n; ++i) {
        diagonal_one[i] = one[(i / 2) * spatial_orbitals + i / 2];
        for (int j = i + 1; j < n; ++j)
            diagonal_two[i * n + j] = diagonal_two[j * n + i] = antisymmetrized(eri,i,j,i,j);
    }
    py::dict result;
    result["n_orb"] = n;
    result["n_elec"] = electrons;
    result["ms2"] = ms2;
    result["core_energy"] = core;
    result["single_offsets"] = array(single_offsets);
    result["single_targets"] = array(single_targets);
    result["single_values"] = array(single_values);
    result["double_offsets"] = array(double_offsets);
    result["double_targets"] = array(double_targets);
    result["double_values"] = array(double_values);
    result["diagonal_one"] = array(diagonal_one);
    result["diagonal_two"] = array(diagonal_two);
    return result;
}
} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("load_tables", &load_tables, pybind11::arg("path"),
          pybind11::arg("spatial_orbitals"), pybind11::arg("electrons"), pybind11::arg("ms2"));
}
