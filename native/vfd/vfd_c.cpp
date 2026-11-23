// Copyright (c) 2026 The Regents of the University of Michigan. All rights reserved.
// Distributed under the BSD 3-Clause license in LICENSE.
#include "vfd_c.h"
#include "vfd.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
using Blocks = std::vector<std::vector<int>>;
constexpr int max_int = std::numeric_limits<int>::max();

void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

template <typename Function>
int32_t guarded(char* error, size_t error_size, Function run) noexcept {
    auto fail = [&](int32_t status, const char* message) {
        if (error && error_size) std::snprintf(error, error_size, "%s", message);
        return status;
    };
    try {
        run();
        return 0;
    } catch (const std::invalid_argument& exc) {
        return fail(1, exc.what());
    } catch (const std::bad_alloc& exc) {
        return fail(2, exc.what());
    } catch (const std::exception& exc) {
        return fail(3, exc.what());
    } catch (...) {
        return fail(3, "Unknown native VFD error.");
    }
}

vfd::MatrixView matrix(int32_t n, const double* data) {
    require(n > 0 && n <= max_int - 63 && data, "Invalid component matrix dimensions.");
    vfd::MatrixView view{data, n};
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row <= col; ++row) {
            const double x = view(row, col), y = view(col, row);
            require(std::isfinite(x) && std::isfinite(y), "Component matrices must be finite.");
            require(std::abs(x - y) <= 1e-12 * (1 + std::abs(x) + std::abs(y)),
                    "Component matrices must be symmetric.");
        }
    }
    return view;
}

Blocks unpack_blocks(int n, int count, const int32_t* offsets, const int32_t* ids,
                     std::vector<int>& membership) {
    require(count > 0 && count <= n && offsets && ids, "Invalid block dimensions.");
    require(offsets[0] == 0 && offsets[count] == n, "Blocks must cover all components.");
    Blocks blocks(count);
    membership.assign(n, -1);
    for (int block = 0; block < count; ++block) {
        const int begin = offsets[block], end = offsets[block + 1];
        require(begin < end && end <= n, "Blocks must be nonempty with ordered offsets.");
        for (int i = begin; i < end; ++i) {
            const int c = ids[i];
            require(c >= 0 && c < n && membership[c] == -1,
                    "Blocks must partition the component IDs exactly once.");
            membership[c] = block;
            blocks[block].push_back(c);
        }
    }
    return blocks;
}
}  // namespace

int32_t asunder_vfd_search_v1(
    int32_t n, const double* B, const double* C, const int64_t* weights,
    size_t pair_count, const int32_t* pairs, int32_t block_count,
    const int32_t* offsets, const int32_t* components, const int64_t* target,
    const VfdOptions* o, int64_t* labels, VfdStats* stats, char* error, size_t error_size) {
    return guarded(error, error_size, [&] {
        const auto B_view = matrix(n, B), C_view = matrix(n, C);
        require(weights && target && o && labels && stats && (!pair_count || pairs),
                "Missing VFD search buffers.");
        require(o->groups > 0 && o->groups <= n && o->lower >= 0 && o->upper >= o->lower &&
                o->upper > 0 && o->restarts >= 1 && o->restarts <= max_int - n &&
                o->local_iters >= 0 && o->local_iters <= (max_int - 7) / 5 &&
                o->tabu_max_steps >= 0 && o->tabu_max_steps <= max_int - 7 &&
                o->shake_rounds >= 0 && std::isfinite(o->w_coassoc), "Invalid VFD search options.");
        int64_t total = 0;
        for (int c = 0; c < n; ++c) {
            require(weights[c] > 0 && weights[c] <= std::numeric_limits<int64_t>::max() - total,
                    "Component weights must be positive with an int64 total.");
            total += weights[c];
        }
        int64_t remaining = total;
        for (int group = 0; group < o->groups; ++group) {
            require(target[group] >= o->lower && target[group] <= o->upper &&
                    target[group] <= remaining, "Invalid target community loads.");
            remaining -= target[group];
        }
        require(remaining == 0, "Target loads must sum to the total component load.");
        std::vector<int> membership;
        auto blocks = unpack_blocks(n, block_count, offsets, components, membership);
        Blocks forbidden(n);
        for (size_t i = 0; i < pair_count; ++i) {
            const int c = pairs[2*i], d = pairs[2*i + 1];
            require(c >= 0 && d >= 0 && c < n && d < n && c != d,
                    "Invalid component cannot-link pair.");
            require(membership[c] != membership[d], "An initial block contains a cannot-link.");
            forbidden[c].push_back(d);
            forbidden[d].push_back(c);
        }
        for (auto& neighbors : forbidden) {
            std::sort(neighbors.begin(), neighbors.end());
            neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
        }
        const vfd::Options options{o->groups, o->lower, o->upper, o->restarts, o->local_iters,
            o->tabu_max_steps, o->shake_rounds, o->w_coassoc, bool(o->allow_block_splitting),
            bool(o->check_invariants), o->seed};
        const auto result = vfd::search(B_view, C_view, {weights, weights + n}, forbidden,
                                        std::move(blocks), {target, target + o->groups}, options);
        std::copy(result.labels.begin(), result.labels.end(), labels);
        *stats = {int32_t(!result.labels.empty()), result.objective_B_sum, result.native_seconds,
            result.restarts_attempted, result.feasible_restarts, result.moves, result.swaps,
            result.ejections, result.splits, result.shake_moves, result.tabu_steps,
            result.candidate_moves, result.candidate_swaps, result.candidate_ejections};
    });
}

int32_t asunder_vfd_action_delta_v1(
    int32_t n, const double* B, const double* C, const int32_t* labels,
    int32_t block_count, const int32_t* offsets, const int32_t* components,
    int32_t first, int32_t second, int32_t destination, double w_coassoc,
    double* changes, char* error, size_t error_size) {
    return guarded(error, error_size, [&] {
        const auto B_view = matrix(n, B), C_view = matrix(n, C);
        require(labels && changes, "Missing diagnostic buffers.");
        std::vector<int> membership;
        const auto blocks = unpack_blocks(n, block_count, offsets, components, membership);
        int groups = 0;
        for (int c = 0; c < n; ++c) {
            require(labels[c] >= 0 && labels[c] < n, "Invalid diagnostic community label.");
            require(labels[c] == labels[blocks[membership[c]].front()],
                    "Diagnostic blocks must lie within one community.");
            groups = std::max(groups, int(labels[c]) + 1);
        }
        require(first >= 0 && first < block_count && second >= -1 && second < block_count &&
                first != second && destination >= 0 && destination < groups &&
                std::isfinite(w_coassoc), "Invalid diagnostic action.");
        const int source = labels[blocks[first].front()];
        if (second < 0) {
            require(source != destination, "A move must change communities.");
        } else {
            const int other = labels[blocks[second].front()];
            require(source != other && other != destination, "The ejected block must leave its community.");
        }
        const auto delta = vfd::diagnose_action(B_view, C_view, {labels, labels + n}, blocks,
                                               first, second, destination, w_coassoc);
        std::copy(delta.begin(), delta.end(), changes);
    });
}
