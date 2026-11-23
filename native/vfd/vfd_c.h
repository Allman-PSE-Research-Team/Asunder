/* C ABI for Asunder's VFD. All buffers are caller-owned for one synchronous call. */
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#define VFD_API __declspec(dllexport)
#else
#define VFD_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int32_t groups;
    int64_t lower, upper;
    int32_t restarts, local_iters, tabu_max_steps, shake_rounds;
    double w_coassoc;
    int32_t allow_block_splitting, check_invariants;
    uint64_t seed;
} VfdOptions;

typedef struct {
    int32_t found;
    double objective_B_sum, native_seconds;
    uint64_t restarts_attempted, feasible_restarts, moves, swaps, ejections, splits;
    uint64_t shake_moves, tabu_steps, candidate_moves, candidate_swaps, candidate_ejections;
} VfdStats;

/* Status: 0 = success (see found), 1 = invalid input, 2 = allocation failure,
 * 3 = internal error. On failure, error receives a NUL-terminated description.
 * Both matrices have n*n doubles in column-major order. Block offsets have
 * block_count+1 entries delimiting n component IDs. Pairs have two IDs each.
 * Weights and output labels have n entries; target has options.groups entries. */
VFD_API int32_t asunder_vfd_search_v1(
    int32_t n, const double* B, const double* C, const int64_t* weights,
    size_t pair_count, const int32_t* pairs, int32_t block_count,
    const int32_t* offsets, const int32_t* components, const int64_t* target,
    const VfdOptions* options, int64_t* labels, VfdStats* stats,
    char* error, size_t error_size);

/* Four output doubles hold the predicted and observed changes to B and C. */
VFD_API int32_t asunder_vfd_action_delta_v1(
    int32_t n, const double* B, const double* C, const int32_t* labels,
    int32_t block_count, const int32_t* offsets, const int32_t* components,
    int32_t first, int32_t second, int32_t destination, double w_coassoc,
    double* changes, char* error, size_t error_size);

#ifdef __cplusplus
}
#endif
