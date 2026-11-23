#pragma once

#include <cstdint>
#include <vector>

namespace vfd {

/** Borrow a column-contiguous square matrix for synchronous native computation.
 *
 * Parameters
 * ----------
 * data : const double*
 *     Array storage retained by the Python caller throughout the search.
 * n : int
 *     Number of rows and columns.
 *
 * Notes
 * -----
 * Matrices must be finite and symmetric. The view never owns or mutates data.
 */
struct MatrixView {
    const double* data;
    int n;

    /** Read one matrix entry.
     *
     * Parameters
     * ----------
     * row, column : int
     *     Zero-based indices within the matrix.
     *
     * Returns
     * -------
     * double
     *     Value at the selected row and column.
     */
    double operator()(int row, int column) const { return data[column * std::size_t(n) + row]; }
};

/** Configure construction and search for one fixed community count.
 *
 * Parameters
 * ----------
 * groups : int
 *     Number of nonempty communities required in the returned solution.
 * lower, upper : int64_t
 *     Inclusive community-load bounds.
 * restarts : int
 *     Constructive restart count; block splitting can add restarts.
 * local_iters, tabu_max_steps : int
 *     Each tabu round uses max(tabu_max_steps, 5 * local_iters) steps.
 * shake_rounds : int
 *     Number of tabu rounds, with feasible shakes between successive rounds.
 * w_coassoc : double
 *     Co-association weight in the search objective, not the final B score.
 * allow_block_splitting : bool
 *     Permit splitting fingerprint blocks into must-link components.
 * check_invariants : bool
 *     Recompute objectives and feasibility after transitions for diagnostics.
 * seed : uint64_t
 *     Seed for the native random engine.
 */
struct Options {
    int groups = 2;
    std::int64_t lower = 1, upper = 1;
    int restarts = 6, local_iters = 60, tabu_max_steps = 60, shake_rounds = 3;
    double w_coassoc = 0.05;
    bool allow_block_splitting = true, check_invariants = false;
    std::uint64_t seed = 42;
};

/** Record native work counters and the best feasible component assignment.
 *
 * Attributes
 * ----------
 * labels : vector of int
 *     Component community IDs; empty when no solution was found.
 * objective_B_sum : double
 *     Sum of W_B entries within returned communities.
 * native_seconds : double
 *     Time inside the native search, including native state construction.
 * restarts_attempted, feasible_restarts : uint64_t
 *     Counts of attempts and successfully constructed feasible assignments.
 * moves, swaps, ejections, splits, shake_moves, tabu_steps : uint64_t
 *     Counts across all restarts, including exploratory actions.
 * candidate_moves, candidate_swaps, candidate_ejections : uint64_t
 *     Numbers of candidate actions examined.
 */
struct Result {
    std::vector<int> labels;
    double objective_B_sum = 0.0, native_seconds = 0.0;
    std::uint64_t restarts_attempted = 0, feasible_restarts = 0;
    std::uint64_t moves = 0, swaps = 0, ejections = 0, splits = 0;
    std::uint64_t shake_moves = 0, tabu_steps = 0;
    std::uint64_t candidate_moves = 0, candidate_swaps = 0, candidate_ejections = 0;
};

/** Run the load-balancing VFD search on prepared component matrices.
 *
 * Parameters
 * ----------
 * B, C : MatrixView
 *     Summed modularity matrix and averaged co-association matrix.
 * weights : vector of int64_t
 *     Positive integral load per must-link component.
 * forbidden : vector of vector of int
 *     Symmetric cannot-link neighbor lists at component level.
 * blocks : vector of vector of int
 *     Initial nonempty, conflict-free blocks covering every component once.
 * target : vector of int64_t
 *     Construction guidance loads with length options.groups.
 * options : Options
 *     Bounds, search budgets, seed, and diagnostic configuration.
 *
 * Returns
 * -------
 * Result
 *     Best B-scored solution across feasible restarts, or empty labels.
 *
 * Notes
 * -----
 * The implementation uses only native containers and does not call Python.
 * Cannot-link masks are arrays of 64-bit words with no 64-component limit.
 */
Result search(MatrixView B, MatrixView C, const std::vector<std::int64_t>& weights,
              const std::vector<std::vector<int>>& forbidden,
              std::vector<std::vector<int>> blocks,
              const std::vector<std::int64_t>& target, const Options& options);

/** Evaluate an action gain and its observed objective change for diagnostics.
 *
 * Parameters
 * ----------
 * B, C : MatrixView
 *     Symmetric component matrices.
 * labels : vector of int
 *     Current community ID per component.
 * blocks : vector of vector of int
 *     Nonempty blocks forming a complete component partition.
 * first, second, destination : int
 *     Block to move, block to eject, and destination of the second block.
 *     For a move, second is -1 and destination is the first block's target.
 * w_coassoc : double
 *     Co-association weight in the combined objective.
 *
 * Returns
 * -------
 * vector of double
 *     Predicted B and C gains followed by recomputed B and C gains.
 */
std::vector<double> diagnose_action(MatrixView B, MatrixView C,
    const std::vector<int>& labels, const std::vector<std::vector<int>>& blocks,
    int first, int second, int destination, double w_coassoc);

}  // namespace vfd
