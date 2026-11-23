// Copyright (c) 2026 The Regents of the University of Michigan. All rights reserved.
// Distributed under the BSD 3-Clause license in LICENSE.
#include "vfd.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <tuple>

namespace vfd {
namespace {
using Load = std::int64_t;
using Mask = std::vector<std::uint64_t>;
using Engine = std::mt19937_64;

/** Store a mutable fingerprint block and its invariant component data.
 *
 * Attributes
 * ----------
 * components : vector of int
 *     Must-link component IDs kept together until a permitted block split.
 * load : int64_t
 *     Total integral load of these components.
 * internal_B, internal_C : double
 *     Complete internal matrix sums, including diagonal entries.
 * membership, forbidden : vector of uint64_t
 *     Component membership and cannot-link masks of arbitrary word count.
 * group : int
 *     Current community, or -1 during construction or an atomic transition.
 */
struct Block {
    std::vector<int> components;
    Load load = 0;
    double internal_B = 0, internal_C = 0;
    Mask membership, forbidden;
    int group = -1;
};

/** Represent one admissible move, swap, or two-block ejection.
 *
 * Attributes
 * ----------
 * kind : int
 *     0 for no action, 1 for a move, 2 for a swap, 3 for an ejection.
 * first, second : int
 *     First and optional second block IDs.
 * source, target, ejected_target : int
 *     Source of the first block, its target, and target of the second block.
 * gain : double
 *     Change in B + w_coassoc * C.
 */
struct Action {
    int kind = 0, first = -1, second = -1, source = -1, target = -1, ejected_target = -1;
    double gain = -std::numeric_limits<double>::infinity();
};

/** Retain the best combined-objective assignment from one tabu invocation.
 *
 * Attributes
 * ----------
 * total, B : double
 *     Combined objective and B objective at the same stored assignment.
 * labels : vector of int
 *     Community ID per component at that assignment.
 */
struct Snapshot {
    double total, B;
    std::vector<int> labels;
};

/** Maintain native state for one constructive restart and its local search.
 *
 * Parameters
 * ----------
 * B, C : MatrixView
 *     Immutable, symmetric component matrices.
 * weights, forbidden : native vectors
 *     Component loads and cannot-link neighbors.
 * initial : vector of vector of int
 *     Blocks that cover every component once.
 * options : Options
 *     Fixed community count, balance bounds, and search configuration.
 * rng : mt19937_64
 *     Random engine shared across constructive restarts.
 * counters : Result
 *     Accumulated native work counters.
 */
class State {
public:
    MatrixView B, C;
    const std::vector<Load>& weights;
    const std::vector<std::vector<int>>& forbidden;
    const Options& options;
    Engine& rng;
    Result& counters;
    int n, k, words;
    Load total_load;
    std::vector<Block> blocks;
    std::vector<int> labels, component_counts;
    std::vector<Load> loads;
    std::vector<std::vector<int>> members;
    std::vector<Mask> masks;
    std::vector<std::vector<double>> sums_B, sums_C;
    std::vector<double> totals_B, totals_C;
    std::vector<std::vector<int>> tabu;

    /** Initialize empty communities and compute immutable block summaries.
     *
     * Parameters
     * ----------
     * Matrix and constraint arguments : native views and vectors
     *     Inputs described in State's class documentation.
     */
    State(MatrixView B_, MatrixView C_, const std::vector<Load>& weights_,
          const std::vector<std::vector<int>>& forbidden_,
          const std::vector<std::vector<int>>& initial, const Options& options_,
          Engine& rng_, Result& counters_)
        : B(B_), C(C_), weights(weights_), forbidden(forbidden_), options(options_),
          rng(rng_), counters(counters_), n(B.n), k(options.groups), words((n + 63) / 64),
          total_load(std::accumulate(weights.begin(), weights.end(), Load(0))),
          labels(n, -1), component_counts(k, 0), loads(k, 0), members(k),
          masks(k, Mask(words, 0)), sums_B(k, std::vector<double>(n, 0)),
          sums_C(k, std::vector<double>(n, 0)), totals_B(k, 0), totals_C(k, 0) {
        blocks.reserve(n);
        tabu.reserve(n);
        for (const auto& components : initial) {
            blocks.push_back(make_block(components));
            tabu.emplace_back(k, -1);
        }
    }

    /** Summarize a block's load, internal scores, and dynamic constraint masks.
     *
     * Parameters
     * ----------
     * components : vector of int
     *     Nonempty collection of distinct component IDs.
     *
     * Returns
     * -------
     * Block
     *     Unassigned block with precomputed immutable summaries.
     */
    Block make_block(const std::vector<int>& components) const {
        Block block;
        block.components = components;
        block.membership.assign(words, 0);
        block.forbidden.assign(words, 0);
        for (int c : components) {
            block.load += weights[c];
            block.membership[c / 64] |= std::uint64_t(1) << (c % 64);
            for (int other : forbidden[c])
                block.forbidden[other / 64] |= std::uint64_t(1) << (other % 64);
            for (int other : components) {
                block.internal_B += B(c, other);
                block.internal_C += C(c, other);
            }
        }
        return block;
    }

    /** Sum one group's cached matrix contributions over a block.
     *
     * Parameters
     * ----------
     * sums : vector of vector of double
     *     B or C column-sum cache indexed by group and component.
     * bi, group : int
     *     Block and community IDs.
     *
     * Returns
     * -------
     * double
     *     Cross-sum between the block and current community components.
     */
    double cross_group(const std::vector<std::vector<double>>& sums, int bi, int group) const {
        double value = 0;
        for (int c : blocks[bi].components) value += sums[group][c];
        return value;
    }

    /** Sum matrix entries between two complete blocks.
     *
     * Parameters
     * ----------
     * M : MatrixView
     *     B or C matrix.
     * bi, bj : int
     *     Block IDs.
     *
     * Returns
     * -------
     * double
     *     Sum of M over the selected Cartesian product.
     */
    double cross_blocks(MatrixView M, int bi, int bj) const {
        double value = 0;
        for (int c : blocks[bi].components)
            for (int d : blocks[bj].components) value += M(c, d);
        return value;
    }

    /** Test cannot-links against a community, optionally excluding one block.
     *
     * Parameters
     * ----------
     * bi, group : int
     *     Incoming block and destination community.
     * removed : int, default=-1
     *     Destination block excluded from the membership test.
     *
     * Returns
     * -------
     * bool
     *     Whether all cannot-links remain satisfied.
     */
    bool conflict_free(int bi, int group, int removed = -1) const {
        for (int w = 0; w < words; ++w) {
            std::uint64_t membership = masks[group][w];
            if (removed >= 0) membership &= ~blocks[removed].membership[w];
            if (blocks[bi].forbidden[w] & membership) return false;
        }
        return true;
    }

    /** Test the upper load bound and cannot-links for direct insertion.
     *
     * Parameters
     * ----------
     * bi, group : int
     *     Incoming block and destination group.
     *
     * Returns
     * -------
     * bool
     *     Whether the insertion respects the upper bound and cannot-links.
     */
    bool can_add(int bi, int group) const {
        return blocks[bi].load <= options.upper - loads[group] && conflict_free(bi, group);
    }

    /** Test final load bounds and cannot-links after replacing a block.
     *
     * Parameters
     * ----------
     * bi, group, removed : int
     *     Incoming block, destination, and outgoing block IDs.
     *
     * Returns
     * -------
     * bool
     *     Whether the final destination community is feasible.
     */
    bool can_replace(int bi, int group, int removed) const {
        Load final_load = loads[group] - blocks[removed].load + blocks[bi].load;
        return options.lower <= final_load && final_load <= options.upper &&
               conflict_free(bi, group, removed);
    }

    /** Insert an unassigned block and update all cached group quantities.
     *
     * Parameters
     * ----------
     * bi, group : int
     *     Block and destination community IDs.
     */
    void add(int bi, int group) {
        Block& block = blocks[bi];
        totals_B[group] += 2 * cross_group(sums_B, bi, group) + block.internal_B;
        totals_C[group] += 2 * cross_group(sums_C, bi, group) + block.internal_C;
        for (int c : block.components) {
            labels[c] = group;
            const double* column_B = B.data + std::size_t(c) * n;
            const double* column_C = C.data + std::size_t(c) * n;
            for (int row = 0; row < n; ++row) {
                sums_B[group][row] += column_B[row];
                sums_C[group][row] += column_C[row];
            }
        }
        block.group = group;
        members[group].push_back(bi);
        loads[group] += block.load;
        component_counts[group] += int(block.components.size());
        for (int w = 0; w < words; ++w) masks[group][w] |= block.membership[w];
    }

    /** Remove an assigned block and update all cached group quantities.
     *
     * Parameters
     * ----------
     * bi, group : int
     *     Block and its current community.
     */
    void remove(int bi, int group) {
        Block& block = blocks[bi];
        totals_B[group] -= 2 * cross_group(sums_B, bi, group) - block.internal_B;
        totals_C[group] -= 2 * cross_group(sums_C, bi, group) - block.internal_C;
        for (int c : block.components) {
            labels[c] = -1;
            const double* column_B = B.data + std::size_t(c) * n;
            const double* column_C = C.data + std::size_t(c) * n;
            for (int row = 0; row < n; ++row) {
                sums_B[group][row] -= column_B[row];
                sums_C[group][row] -= column_C[row];
            }
        }
        block.group = -1;
        auto& group_members = members[group];
        group_members.erase(std::find(group_members.begin(), group_members.end(), bi));
        loads[group] -= block.load;
        component_counts[group] -= int(block.components.size());
        for (int w = 0; w < words; ++w) masks[group][w] &= ~block.membership[w];
    }

    /** Compute one community's average off-diagonal co-association.
     *
     * Parameters
     * ----------
     * group : int
     *     Community ID.
     *
     * Returns
     * -------
     * double
     *     Average pair cohesion, or zero for at most one component.
     */
    double cohesion(int group) const {
        int count = component_counts[group];
        return count <= 1 ? 0 : (totals_C[group] - count) / (double(count) * (count - 1));
    }

    /** Compute a block's average attachment to other components in its group.
     *
     * Parameters
     * ----------
     * bi, group : int
     *     Assigned block and its community.
     *
     * Returns
     * -------
     * double
     *     Mean co-association between the block and the rest of its community.
     */
    double attachment(int bi, int group) const {
        const Block& block = blocks[bi];
        int remaining = std::max(1, component_counts[group] - int(block.components.size()));
        return (cross_group(sums_C, bi, group) - block.internal_C) /
               std::max(1.0, double(block.components.size()) * remaining);
    }

    /** Order communities by increasing cohesion with stable ties.
     *
     * Returns
     * -------
     * vector of int
     *     All community IDs in exploration order.
     */
    std::vector<int> group_order() const {
        std::vector<int> order(k);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int x, int y) { return cohesion(x) < cohesion(y); });
        return order;
    }

    /** Select weakly attached blocks from a community.
     *
     * Parameters
     * ----------
     * group, limit : int
     *     Community ID and maximum returned candidate count.
     * shuffled : bool, default=False
     *     Shuffle before sorting to randomize equal attachment and load ties.
     *
     * Returns
     * -------
     * vector of int
     *     Candidate block IDs sorted by attachment and integral load.
     */
    std::vector<int> candidates(int group, int limit, bool shuffled = false) {
        auto candidate = members[group];
        if (shuffled) std::shuffle(candidate.begin(), candidate.end(), rng);
        std::vector<double> attachments(blocks.size(), 0);
        for (int bi : candidate) attachments[bi] = attachment(bi, group);
        std::stable_sort(candidate.begin(), candidate.end(), [&](int x, int y) {
            return std::tie(attachments[x], blocks[x].load) < std::tie(attachments[y], blocks[y].load);
        });
        if (int(candidate.size()) > limit) candidate.resize(limit);
        return candidate;
    }

    /** Split a block into singleton must-link components.
     *
     * Parameters
     * ----------
     * bi : int
     *     Block ID to split. An assigned block retains its community.
     *
     * Returns
     * -------
     * bool
     *     Whether a multi-component block was split.
     */
    bool split(int bi) {
        if (!options.allow_block_splitting || blocks[bi].components.size() <= 1) return false;
        auto components = blocks[bi].components;
        int group = blocks[bi].group;
        blocks[bi] = make_block({components[0]});
        blocks[bi].group = group;
        std::vector<int> appended;
        for (std::size_t i = 1; i < components.size(); ++i) {
            appended.push_back(int(blocks.size()));
            blocks.push_back(make_block({components[i]}));
            blocks.back().group = group;
            tabu.emplace_back(k, -1);
        }
        if (group >= 0) {
            auto& group_members = members[group];
            auto position = std::find(group_members.begin(), group_members.end(), bi);
            group_members.insert(position + 1, appended.begin(), appended.end());
        }
        ++counters.splits;
        return true;
    }

    /** Export the current block decomposition for a constructive retry.
     *
     * Returns
     * -------
     * vector of vector of int
     *     Copy of every block's component IDs.
     */
    std::vector<std::vector<int>> block_components() const {
        std::vector<std::vector<int>> out;
        for (const auto& block : blocks) out.push_back(block.components);
        return out;
    }

    /** Seed every community and greedily assign the remaining blocks.
     *
     * Parameters
     * ----------
     * target : vector of int64_t
     *     Guidance load per community.
     * split_retry : bool&
     *     Set when construction failed after splitting a coarse block.
     *
     * Returns
     * -------
     * bool
     *     Whether all components were assigned before lower-bound repair.
     */
    bool construct(const std::vector<Load>& target, bool& split_retry) {
        split_retry = false;
        if (int(blocks.size()) < k) return false;
        std::vector<int> order(blocks.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int x, int y) {
            return std::tie(blocks[x].load, x) > std::tie(blocks[y].load, y);
        });
        int seeded = 0;
        for (int bi : order) {
            if (blocks[bi].load > options.upper) continue;
            add(bi, seeded++);
            if (seeded == k) break;
        }
        if (seeded != k) return false;
        for (int bi : order) {
            if (blocks[bi].group >= 0) continue;
            Load remaining = total_load - std::accumulate(loads.begin(), loads.end(), Load(0));
            Load deficit = 0;
            for (Load load : loads) deficit += std::max(Load(0), options.lower - load);
            int best_group = -1;
            std::tuple<int, double, Load, double, int> best_score;
            for (int group = 0; group < k; ++group) {
                if (!can_add(bi, group)) continue;
                Load old_deficit = std::max(Load(0), options.lower - loads[group]);
                Load new_deficit = std::max(Load(0), options.lower - loads[group] - blocks[bi].load);
                if (deficit - old_deficit + new_deficit > remaining - blocks[bi].load) continue;
                double gain = 2 * cross_group(sums_B, bi, group) +
                              options.w_coassoc * cross_group(sums_C, bi, group);
                auto score = std::make_tuple(loads[group] < options.lower ? 0 : 1, -gain,
                    std::abs(loads[group] + blocks[bi].load - target[group]),
                    std::generate_canonical<double, 53>(rng), group);
                if (best_group < 0 || score < best_score) {
                    best_score = score;
                    best_group = group;
                }
            }
            if (best_group < 0) {
                split_retry = split(bi);
                return false;
            }
            add(bi, best_group);
        }
        return true;
    }

    /** Repair lower-load deficits using low-cohesion donor communities.
     *
     * Returns
     * -------
     * bool
     *     Whether all lower bounds were achieved within 20,000 repair steps.
     */
    bool repair() {
        for (int step = 0; step < 20000; ++step) {
            int need = -1;
            for (int group = 0; group < k; ++group)
                if (loads[group] < options.lower && (need < 0 || loads[group] < loads[need])) need = group;
            if (need < 0) return true;
            bool moved = false;
            int splittable = -1;
            for (int from : group_order()) {
                if (loads[from] <= options.lower) continue;
                for (int bi : candidates(from, int(blocks.size()), true)) {
                    if (loads[from] - blocks[bi].load < std::max(Load(1), options.lower) || !can_add(bi, need)) {
                        if (splittable < 0 && options.allow_block_splitting && blocks[bi].components.size() > 1)
                            splittable = bi;
                        continue;
                    }
                    remove(bi, from);
                    add(bi, need);
                    moved = true;
                    break;
                }
                if (moved) break;
            }
            if (!moved && (splittable < 0 || !split(splittable))) return false;
        }
        return false;
    }

    /** Compute the exact matrix gain of a single-block move.
     *
     * Parameters
     * ----------
     * sums : native column cache
     *     B or C group sums.
     * bi, source, target : int
     *     Moved block and its source and destination communities.
     * is_B : bool
     *     Select the block's B internal sum instead of its C internal sum.
     *
     * Returns
     * -------
     * double
     *     Change in the complete within-community matrix sum.
     */
    double move_gain(const std::vector<std::vector<double>>& sums, int bi,
                     int source, int target, bool is_B) const {
        double internal = is_B ? blocks[bi].internal_B : blocks[bi].internal_C;
        return 2 * (cross_group(sums, bi, target) - cross_group(sums, bi, source) + internal);
    }

    /** Compute a two-block action gain using its cross-block correction.
     *
     * Parameters
     * ----------
     * bi, bj, source, target, ejected_target : int
     *     First block moves source to target; second moves target to ejected_target.
     * is_B : bool
     *     Select modularity or co-association matrix contributions.
     *
     * Returns
     * -------
     * double
     *     Exact matrix-sum change, including the swap correction when targets coincide.
     */
    double ejection_gain(int bi, int bj, int source, int target, int ejected_target, bool is_B) const {
        const auto& sums = is_B ? sums_B : sums_C;
        double internal = is_B ? blocks[bi].internal_B : blocks[bi].internal_C;
        double cross = cross_blocks(is_B ? B : C, bi, bj);
        double incoming = cross_group(sums, bi, target) - cross;
        double outgoing = cross_group(sums, bi, source) + (ejected_target == source ? cross : 0);
        return move_gain(sums, bj, target, ejected_target, is_B) +
               2 * (incoming - outgoing + internal);
    }

    /** Return the current unnormalized modularity-matrix score.
     *
     * Returns
     * -------
     * double
     *     Sum of cached B totals across communities.
     */
    double B_score() const { return std::accumulate(totals_B.begin(), totals_B.end(), 0.0); }

    /** Return the combined search score.
     *
     * Returns
     * -------
     * double
     *     B score plus w_coassoc times the complete C within-group sum.
     */
    double total_score() const {
        return B_score() + options.w_coassoc * std::accumulate(totals_C.begin(), totals_C.end(), 0.0);
    }

    /** Check a block-destination tabu expiry.
     *
     * Parameters
     * ----------
     * bi, group, step : int
     *     Block, forbidden destination, and current tabu step.
     *
     * Returns
     * -------
     * bool
     *     Whether the destination is still tabu.
     */
    bool is_tabu(int bi, int group, int step) const { return tabu[bi][group] > step; }

    /** Apply an atomic move, swap, or ejection and optionally audit its state.
     *
     * Parameters
     * ----------
     * action : Action
     *     Feasible action evaluated against the current state.
     */
    void apply(const Action& action) {
        double previous = total_score();
        if (action.kind == 1) {
            remove(action.first, action.source);
            add(action.first, action.target);
            ++counters.moves;
        } else if (action.kind == 2) {
            remove(action.first, action.source);
            remove(action.second, action.target);
            add(action.first, action.target);
            add(action.second, action.source);
            ++counters.swaps;
        } else {
            remove(action.second, action.target);
            add(action.second, action.ejected_target);
            remove(action.first, action.source);
            add(action.first, action.target);
            ++counters.ejections;
        }
        if (options.check_invariants) {
            audit();
            if (std::abs(total_score() - previous - action.gain) >
                1e-8 * (1 + std::abs(previous) + std::abs(action.gain)))
                throw std::logic_error("Native action gain differs from the observed objective change.");
        }
    }

    /** Verify cached scores, membership, loads, and cannot-link feasibility.
     *
     * Raises
     * ------
     * logic_error
     *     If an incremental cache or constraint differs from full recomputation.
     */
    void audit() const {
        std::vector<int> seen(n, 0), counts(k, 0);
        std::vector<Load> expected_loads(k, 0);
        std::vector<double> expected_B(k, 0), expected_C(k, 0);
        for (int group = 0; group < k; ++group) {
            for (int bi : members[group]) {
                if (blocks[bi].group != group) throw std::logic_error("Block membership cache mismatch.");
                for (int c : blocks[bi].components) {
                    if (labels[c] != group || ++seen[c] != 1) throw std::logic_error("Component membership mismatch.");
                }
            }
        }
        for (int c = 0; c < n; ++c) {
            int group = labels[c];
            if (group < 0 || group >= k || seen[c] != 1) throw std::logic_error("Invalid component assignment.");
            ++counts[group];
            expected_loads[group] += weights[c];
            if (!(masks[group][c / 64] & (std::uint64_t(1) << (c % 64))))
                throw std::logic_error("Dynamic membership mask mismatch.");
            for (int other : forbidden[c])
                if (labels[other] == group) throw std::logic_error("Cannot-link violation.");
            for (int d = 0; d < n; ++d) {
                if (labels[d] == group) {
                    expected_B[group] += B(c, d);
                    expected_C[group] += C(c, d);
                }
            }
        }
        for (int group = 0; group < k; ++group) {
            if (counts[group] != component_counts[group] || expected_loads[group] != loads[group])
                throw std::logic_error("Group load or component-count cache mismatch.");
            if (counts[group] == 0 || loads[group] < options.lower || loads[group] > options.upper)
                throw std::logic_error("Community load-bound violation.");
            if (std::abs(expected_B[group] - totals_B[group]) > 1e-8 * (1 + std::abs(expected_B[group])) ||
                std::abs(expected_C[group] - totals_C[group]) > 1e-8 * (1 + std::abs(expected_C[group])))
                throw std::logic_error("Cached group objective mismatch.");
            for (int row = 0; row < n; ++row) {
                double expected_sum_B = 0, expected_sum_C = 0;
                for (int col = 0; col < n; ++col) if (labels[col] == group) {
                    expected_sum_B += B(row, col);
                    expected_sum_C += C(row, col);
                }
                if (std::abs(expected_sum_B - sums_B[group][row]) > 1e-8 * (1 + std::abs(expected_sum_B)) ||
                    std::abs(expected_sum_C - sums_C[group][row]) > 1e-8 * (1 + std::abs(expected_sum_C)))
                    throw std::logic_error("Cached group column-sum mismatch.");
            }
        }
    }

    /** Search moves, swaps, and ejections under tabu restrictions and aspiration.
     *
     * Parameters
     * ----------
     * max_steps : int
     *     Maximum tabu steps. Early termination occurs after prolonged stagnation.
     *
     * Returns
     * -------
     * Snapshot
     *     Best combined-score assignment encountered in this invocation.
     */
    Snapshot improve(int max_steps) {
        Snapshot best{total_score(), B_score(), labels};
        int no_improvement = 0;
        int group_limit = std::min(k, 6);
        for (int step = 0; step < max_steps; ++step) {
            ++counters.tabu_steps;
            double base = total_score();
            auto order = group_order();
            Action best_move, best_swap, best_ejection;
            for (int from : order) {
                for (int bi : candidates(from, 10)) {
                    if (loads[from] - blocks[bi].load < std::max(Load(1), options.lower)) continue;
                    std::vector<int> destinations(k);
                    std::iota(destinations.begin(), destinations.end(), 0);
                    std::shuffle(destinations.begin(), destinations.end(), rng);
                    for (int j = 0; j < group_limit; ++j) {
                        int to = destinations[j];
                        if (to == from) continue;
                        ++counters.candidate_moves;
                        if (!can_add(bi, to)) continue;
                        double gain = move_gain(sums_B, bi, from, to, true) +
                                      options.w_coassoc * move_gain(sums_C, bi, from, to, false);
                        if (is_tabu(bi, to, step) && base + gain <= best.total) continue;
                        if (gain > best_move.gain) best_move = {1, bi, -1, from, to, -1, gain};
                    }
                }
            }
            for (int j = 0; j < std::max(1, k / 2); ++j) {
                int g1 = order[j];
                std::vector<int> others(k);
                std::iota(others.begin(), others.end(), 0);
                std::shuffle(others.begin(), others.end(), rng);
                for (int pos = 0; pos < std::min(k, 4); ++pos) {
                    int g2 = others[pos];
                    if (g1 == g2 || members[g1].empty() || members[g2].empty()) continue;
                    auto first = candidates(g1, 10), second = candidates(g2, 10);
                    for (int bi : first) for (int bj : second) {
                        ++counters.candidate_swaps;
                        if (!can_replace(bi, g2, bj) || !can_replace(bj, g1, bi)) continue;
                        double gain = ejection_gain(bi, bj, g1, g2, g1, true) +
                                      options.w_coassoc * ejection_gain(bi, bj, g1, g2, g1, false);
                        if ((is_tabu(bi, g2, step) || is_tabu(bj, g1, step)) && base + gain <= best.total) continue;
                        if (gain > best_swap.gain) best_swap = {2, bi, bj, g1, g2, g1, gain};
                    }
                }
            }
            for (int from : order) {
                for (int bi : candidates(from, 10)) {
                    if (loads[from] - blocks[bi].load < std::max(Load(1), options.lower)) continue;
                    std::vector<int> destinations(k);
                    std::iota(destinations.begin(), destinations.end(), 0);
                    std::shuffle(destinations.begin(), destinations.end(), rng);
                    for (int pos = 0; pos < group_limit; ++pos) {
                        int to = destinations[pos];
                        if (to == from || can_add(bi, to) || members[to].empty()) continue;
                        for (int bj : candidates(to, 10)) {
                            if (!can_replace(bi, to, bj)) continue;
                            std::vector<int> ejected_destinations(k);
                            std::iota(ejected_destinations.begin(), ejected_destinations.end(), 0);
                            std::shuffle(ejected_destinations.begin(), ejected_destinations.end(), rng);
                            for (int gk : ejected_destinations) {
                                if (gk == to) continue;
                                ++counters.candidate_ejections;
                                // Preserve the conservative pre-insertion capacity check.
                                if (blocks[bj].load > options.upper - loads[gk]) continue;
                                Load final_from = loads[from] - blocks[bi].load + (gk == from ? blocks[bj].load : 0);
                                if (final_from < std::max(Load(1), options.lower)) continue;
                                if (!conflict_free(bj, gk, gk == from ? bi : -1)) continue;
                                double gain = ejection_gain(bi, bj, from, to, gk, true) +
                                              options.w_coassoc * ejection_gain(bi, bj, from, to, gk, false);
                                if ((is_tabu(bi, to, step) || is_tabu(bj, gk, step)) && base + gain <= best.total) continue;
                                if (gain > best_ejection.gain) best_ejection = {3, bi, bj, from, to, gk, gain};
                            }
                        }
                    }
                }
            }
            Action chosen;
            for (const auto& candidate : {best_move, best_swap, best_ejection})
                if (candidate.kind && candidate.gain > chosen.gain) chosen = candidate;
            if (!chosen.kind) {
                bool did_split = false;
                for (int group : order) {
                    std::vector<int> multi;
                    for (int bi : members[group]) if (blocks[bi].components.size() > 1) multi.push_back(bi);
                    std::stable_sort(multi.begin(), multi.end(), [&](int x, int y) { return attachment(x, group) < attachment(y, group); });
                    if (!multi.empty() && split(multi[0])) {
                        did_split = true;
                        if (options.check_invariants) audit();
                        break;
                    }
                }
                if (did_split) continue;
                break;
            }
            apply(chosen);
            tabu[chosen.first][chosen.source] = step + 7;
            if (chosen.kind == 2) tabu[chosen.second][chosen.target] = step + 7;
            if (chosen.kind == 3) tabu[chosen.second][chosen.target] = step + 7;
            double current = total_score();
            if (current > best.total + 1e-12) {
                best = {current, B_score(), labels};
                no_improvement = 0;
            } else if (++no_improvement > max_steps / 4) break;
        }
        return best;
    }

    /** Perturb the current assignment with a bounded number of feasible moves.
     *
     * Parameters
     * ----------
     * move_limit : int
     *     Maximum number of shake moves.
     */
    void shake(int move_limit) {
        int moved = 0;
        for (int from : group_order()) {
            for (int bi : candidates(from, 20)) {
                if (moved >= move_limit) return;
                if (loads[from] - blocks[bi].load < std::max(Load(1), options.lower)) continue;
                std::vector<int> destinations(k);
                std::iota(destinations.begin(), destinations.end(), 0);
                std::shuffle(destinations.begin(), destinations.end(), rng);
                for (int to : destinations) {
                    if (to == from || !can_add(bi, to)) continue;
                    double gain = move_gain(sums_B, bi, from, to, true) +
                                  options.w_coassoc * move_gain(sums_C, bi, from, to, false);
                    apply({1, bi, -1, from, to, -1, gain});
                    ++counters.shake_moves;
                    ++moved;
                    break;
                }
            }
        }
    }
};

/** Recompute a complete within-community matrix sum from labels.
 *
 * Parameters
 * ----------
 * matrix : MatrixView
 *     Square component matrix.
 * labels : vector of int
 *     Assigned community IDs for every component.
 *
 * Returns
 * -------
 * double
 *     Sum over all component pairs sharing a community.
 */
double score(MatrixView matrix, const std::vector<int>& labels) {
    double total = 0;
    for (int c = 0; c < matrix.n; ++c)
        for (int d = 0; d < matrix.n; ++d)
            if (labels[c] == labels[d]) total += matrix(c, d);
    return total;
}

/** Validate nonemptiness, balance, and cannot-links for a stored assignment.
 *
 * Parameters
 * ----------
 * labels, weights, forbidden : native vectors
 *     Assignment and component-level constraints.
 * options : Options
 *     Required number of groups and inclusive load bounds.
 *
 * Returns
 * -------
 * bool
 *     Whether the stored assignment satisfies every constraint.
 */
bool feasible(const std::vector<int>& labels, const std::vector<Load>& weights,
              const std::vector<std::vector<int>>& forbidden, const Options& options) {
    std::vector<Load> loads(options.groups, 0);
    std::vector<int> counts(options.groups, 0);
    for (int c = 0; c < int(labels.size()); ++c) {
        int group = labels[c];
        if (group < 0 || group >= options.groups) return false;
        loads[group] += weights[c];
        ++counts[group];
        for (int d : forbidden[c]) if (labels[d] == group) return false;
    }
    for (int group = 0; group < options.groups; ++group)
        if (!counts[group] || loads[group] < options.lower || loads[group] > options.upper) return false;
    return true;
}
}  // namespace

Result search(MatrixView B, MatrixView C, const std::vector<Load>& weights,
              const std::vector<std::vector<int>>& forbidden,
              std::vector<std::vector<int>> base_blocks,
              const std::vector<Load>& target, const Options& options) {
    auto started = std::chrono::steady_clock::now();
    Result result;
    Engine rng(options.seed);
    int restart_limit = options.restarts;
    double best_B = -std::numeric_limits<double>::infinity();
    for (int restart = 0; restart < restart_limit; ++restart) {
        ++result.restarts_attempted;
        State state(B, C, weights, forbidden, base_blocks, options, rng, result);
        bool split_retry = false;
        if (!state.construct(target, split_retry)) {
            if (split_retry) {
                base_blocks = state.block_components();
                ++restart_limit;
            }
            continue;
        }
        if (!state.repair()) continue;
        ++result.feasible_restarts;
        if (options.check_invariants) state.audit();
        auto local_labels = state.labels;
        double local_B = state.B_score();
        int shake_moves = std::max(2, int(state.blocks.size()) / 25);
        int max_steps = std::max(options.tabu_max_steps, options.local_iters * 5);
        for (int round = 0; round < options.shake_rounds; ++round) {
            Snapshot best = state.improve(max_steps);
            if (best.B > local_B + 1e-12) {
                local_B = best.B;
                local_labels = std::move(best.labels);
            }
            if (round + 1 < options.shake_rounds) state.shake(shake_moves);
        }
        if (!feasible(local_labels, weights, forbidden, options)) continue;
        double objective = score(B, local_labels);
        if (!std::isfinite(objective)) continue;
        if (objective > best_B) {
            best_B = objective;
            result.objective_B_sum = objective;
            result.labels = std::move(local_labels);
        }
    }
    result.native_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return result;
}

std::vector<double> diagnose_action(MatrixView B, MatrixView C,
    const std::vector<int>& labels, const std::vector<std::vector<int>>& blocks,
    int first, int second, int destination, double w_coassoc) {
    int k = *std::max_element(labels.begin(), labels.end()) + 1;
    Options options;
    options.groups = k;
    options.lower = 0;
    options.upper = B.n;
    options.w_coassoc = w_coassoc;
    std::vector<Load> weights(B.n, 1);
    std::vector<std::vector<int>> forbidden(B.n);
    Engine rng(42);
    Result counters;
    State state(B, C, weights, forbidden, blocks, options, rng, counters);
    for (int bi = 0; bi < int(blocks.size()); ++bi) state.add(bi, labels[blocks[bi][0]]);
    int from = state.blocks[first].group;
    double before_B = score(B, labels), before_C = score(C, labels), predicted_B, predicted_C;
    if (second < 0) {
        predicted_B = state.move_gain(state.sums_B, first, from, destination, true);
        predicted_C = state.move_gain(state.sums_C, first, from, destination, false);
        state.apply({1, first, -1, from, destination, -1, predicted_B + w_coassoc * predicted_C});
    } else {
        int to = state.blocks[second].group;
        predicted_B = state.ejection_gain(first, second, from, to, destination, true);
        predicted_C = state.ejection_gain(first, second, from, to, destination, false);
        state.apply({3, first, second, from, to, destination, predicted_B + w_coassoc * predicted_C});
    }
    return {predicted_B, predicted_C, score(B, state.labels) - before_B, score(C, state.labels) - before_C};
}
}  // namespace vfd
