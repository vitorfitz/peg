#pragma once
#include <vector>
#include <unordered_set>
#include <deque>
#include <pybind11/numpy.h>

class RetrogradeSolver
{
public:
    RetrogradeSolver(const std::vector<std::vector<int>> &graph, int n_pursuers, const std::vector<int> &escape_nodes);
    void run();

    // Exportable views for Python
    pybind11::array_t<short> get_results(int turn);
    pybind11::array_t<int> get_best_moves(int turn);

private:
    int n_nodes, n_pursuers;
    int total_combos, total_positions;
    std::vector<std::vector<int>> graph, adj;
    std::unordered_set<int> escape_nodes;

    std::vector<std::vector<short>> results;
    std::vector<std::vector<short>> degree;
    std::vector<std::vector<int>> best_moves;

    int rankCombo(const std::vector<int> &combo);
    std::vector<int> unrankCombo(int rank);
    int posToIndex(const std::pair<int, std::vector<int>> &pos);
    std::pair<int, std::vector<int>> indexToPos(int idx);
};