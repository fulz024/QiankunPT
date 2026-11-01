#pragma once
#include <cmath>

// Inclusive positive cutoff, shared by materialized and streaming PT.
// Exact zeros remain absent when screening is disabled; they add no amplitude.
__device__ __forceinline__ bool keep_pt_coupling(double value, double cutoff) {
    const double magnitude = fabs(value);
    return magnitude >= cutoff && magnitude > 0.0;
}
