#include "calc.hpp"
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <deque>
#include <set>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <cassert>
#include <chrono>
#include <iomanip>

namespace py = pybind11;
using namespace std;

using vi = vector<int>;
using vvi = vector<vi>;
using pii = pair<int, int>;
using State = pair<int, vector<int>>;

static int comb(int n, int k)
{
    if (k < 0 || k > n) return 0;
    int res = 1;
    for (int i = 1; i <= k; ++i)
        res = res * (n - i + 1) / i;
    return res;
}

RetrogradeSolver::RetrogradeSolver(const std::vector<std::vector<int>> &graph_, int n_pursuers_, const std::vector<int> &escape_nodes_)
    : n_nodes(graph_.size()),
      n_pursuers(n_pursuers_),
      total_combos(0),
      total_positions(0),
      graph(graph_),
      adj(),
      escape_nodes(escape_nodes_.begin(), escape_nodes_.end())
{
    adj.resize(n_nodes);
    for (int i = 0; i < n_nodes; ++i)
    {
        adj[i] = graph[i];
        adj[i].push_back(i);
    }

    total_combos = comb(n_nodes + n_pursuers - 1, n_pursuers);
    total_positions = total_combos * n_nodes;

    results.assign(2, std::vector<short>(total_positions, 0));
    degree.assign(2, std::vector<short>(total_positions, 0));
    best_moves.assign(2, std::vector<int>(total_positions, -1));
}

py::array_t<short> RetrogradeSolver::get_results(int turn)
{
    return py::array_t<short>(
        {total_positions}, {sizeof(short)}, results[turn].data());
}

py::array_t<int> RetrogradeSolver::get_best_moves(int turn)
{
    return py::array_t<int>(
        {total_positions}, {sizeof(int)}, best_moves[turn].data());
}

void RetrogradeSolver::run()
{
    deque<tuple<State, int, int>> q;
    int idx = 0;
    int pre_processed = 0;
    auto t_start = chrono::high_resolution_clock::now();

    cout << "Generating terminal states and degrees" << endl;

    for (int evader = 0; evader < n_nodes; ++evader) {
        vector<int> pursuers(n_pursuers);
        function<void(int, int)> gen = [&](int start, int depth) {
            if (depth == n_pursuers) {
                if (find(pursuers.begin(), pursuers.end(), evader) != pursuers.end()) {
                    results[0][idx] = 1;
                    results[1][idx] = 1;
                    q.push_back({{evader, pursuers}, 0, idx});
                    pre_processed += 2;
                } else {
                    set<vector<int>> next_pursuers;
                    for (int i = 0; i < n_pursuers; ++i) {
                        vector<vector<int>> moves;
                        for (int p : adj[pursuers[i]]) {
                            moves.push_back({p});
                        }
                        if (i == 0)
                            next_pursuers = set<vector<int>>(moves.begin(), moves.end());
                        else {
                            set<vector<int>> temp;
                            for (const auto& base : next_pursuers) {
                                for (const auto& move : moves) {
                                    auto new_combo = base;
                                    new_combo.insert(new_combo.end(), move.begin(), move.end());
                                    sort(new_combo.begin(), new_combo.end());
                                    temp.insert(new_combo);
                                }
                            }
                            next_pursuers = temp;
                        }
                    }
                    degree[1][idx] = next_pursuers.size();

                    if (escape_nodes.count(evader)) {
                        results[0][idx] = -1;
                        q.push_back({{evader, pursuers}, 0, idx});
                        pre_processed++;
                    } else {
                        degree[0][idx] = count_if(adj[evader].begin(), adj[evader].end(),
                                                    [&](int a) { return find(pursuers.begin(), pursuers.end(), a) == pursuers.end(); });
                    }
                }
                idx++;
                return;
            }

            for (int i = start; i < n_nodes; ++i) {
                pursuers[depth] = i;
                gen(i, depth + 1);
            }
        };
        gen(0, 0);
        cout << (evader+1) << "/" << n_nodes << endl;
    }

    int states_to_process = 2 * total_positions - pre_processed; // Set this correctly
    int processed = 0;
    int log_interval = 100000;

    cout << "Retrograde propagation starting" << endl;

    while (!q.empty()){
        auto [pos, turn, idx] = q.front(); q.pop_front();
        int val = results[turn][idx];
        int evader = pos.first;
        vector<int>& pursuers = pos.second;

        if (turn == 0) {
            // Evader just moved: pursuers' turn
            function<void(int, vector<int>&)> generateMoves = [&](int depth, vector<int>& current) {
                if (depth == n_pursuers) {
                    auto sorted = current;
                    sort(sorted.begin(), sorted.end());
                    State prev = {evader, sorted};
                    int prev_idx = posToIndex(prev);
                    if (results[1][prev_idx] == 0) {
                        if (val > 0) {
                            results[1][prev_idx] = val + 1;
                            best_moves[1][prev_idx] = idx;
                            q.push_back({prev, 1, prev_idx});
                            processed++;
                            if (processed % log_interval == 0)
                            {
                                cout << processed << "/" << states_to_process << endl;
                            }
                        } else if (val < 0) {
                            degree[1][prev_idx]--;
                            if (degree[1][prev_idx] == 0) {
                                results[1][prev_idx] = val - 1;
                                best_moves[1][prev_idx] = idx;
                                q.push_back({prev, 1, prev_idx});
                                processed++;
                                if (processed % log_interval == 0)
                                {
                                    cout << processed << "/" << states_to_process << endl;
                                }
                            }
                        }
                    }
                    return;
                }
                for (int p : adj[pursuers[depth]]) {
                    current[depth] = p;
                    generateMoves(depth + 1, current);
                }
            };
            vector<int> current(n_pursuers);
            generateMoves(0, current);

        } else {
            // Pursuers just moved: evader's turn
            for (int e_prev : adj[evader]) {
                if (find(pursuers.begin(), pursuers.end(), e_prev) != pursuers.end()) continue;
                State prev = {e_prev, pursuers};
                int prev_idx = posToIndex(prev);
                if (results[0][prev_idx] == 0) {
                    if (val < 0) {
                        results[0][prev_idx] = val - 1;
                        best_moves[0][prev_idx] = idx;
                        q.push_back({prev, 0, prev_idx});
                        processed++;
                        if (processed % log_interval == 0)
                        {
                            cout << processed << "/" << states_to_process << endl;
                        }
                    } else if (val > 0) {
                        degree[0][prev_idx]--;
                        if (degree[0][prev_idx] == 0) {
                            results[0][prev_idx] = val + 1;
                            best_moves[0][prev_idx] = idx;
                            q.push_back({prev, 0, prev_idx});
                            processed++;
                            if (processed % log_interval == 0)
                            {
                                cout << processed << "/" << states_to_process << endl;
                            }
                        }
                    }
                }
            }
        }
    }

    int stalemates = states_to_process - processed;
    cout << stalemates << " stalemate states" << endl;

    // Stalemates: pick random valid non-losing moves
    for (int turn = 0; turn < 2; ++turn) {
        for (int idx = 0; idx < total_positions; ++idx) {
            if (results[turn][idx] != 0 || best_moves[turn][idx] != -1) continue;

            State pos = indexToPos(idx);
            int evader = pos.first;
            vector<int>& pursuers = pos.second;

            if (turn == 0) {
                for (int e : adj[evader]) {
                    if (find(pursuers.begin(), pursuers.end(), e) != pursuers.end()) continue;
                    int next_idx = posToIndex({e, pursuers});
                    if (results[1][next_idx] <= 0) {
                        best_moves[0][idx] = next_idx;
                        break;
                    }
                }
            } else {
                function<void(int, vector<int>&)> genMoves = [&](int depth, vector<int>& current) {
                    if (depth == n_pursuers) {
                        auto sorted = current;
                        sort(sorted.begin(), sorted.end());
                        int next_idx = posToIndex({evader, sorted});
                        if (results[0][next_idx] >= 0) {
                            best_moves[1][idx] = next_idx;
                        }
                        return;
                    }
                    for (int p : adj[pursuers[depth]]) {
                        current[depth] = p;
                        genMoves(depth + 1, current);
                    }
                };
                vector<int> current(n_pursuers);
                genMoves(0, current);
            }
        }
    }

    auto t_end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(t_end - t_start);

    auto total_s = duration.count();
    auto hours = total_s / 3600;
    total_s %= 3600;
    auto minutes = total_s / 60;
    total_s %= 60;
    auto seconds = total_s;

    cout << "Elapsed: " << setfill('0')
              << hours << ":"
              << setw(2) << minutes << ":"
              << setw(2) << seconds << endl;
}

int RetrogradeSolver::rankCombo(const vector<int> &combo)
{
    int k = combo.size();
    int rank = 0;
    for (int i = 0; i < k; ++i)
        rank += comb(n_nodes - combo[i] + k - i - 2, k - i);
    return total_combos - rank - 1;
}

int RetrogradeSolver::posToIndex(const State &pos)
{
    return pos.first * total_combos + rankCombo(pos.second);
}

vector<int> RetrogradeSolver::unrankCombo(int rank)
{
    rank = total_combos - rank - 1;
    int lex_rank = total_combos - rank - 1;
    vector<int> combo(n_pursuers);
    int x = 0;
    for (int i = 0; i < n_pursuers; ++i) {
        while (true) {
            int c = comb(n_nodes - x + n_pursuers - i - 2, n_pursuers - i - 1);
            if (c <= lex_rank) {
                lex_rank -= c;
                x++;
            } else {
                combo.push_back(x);
                break;
            }
        }
    }
    return combo;
}

State RetrogradeSolver::indexToPos(int idx)
{
    int evader = idx / total_combos;
    int rank = idx % total_combos;
    return {evader, unrankCombo(rank)};
}