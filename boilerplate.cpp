#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "calc.hpp"

namespace py = pybind11;

PYBIND11_MODULE(retrograde_cpp, m)
{
    py::class_<RetrogradeSolver>(m, "RetrogradeSolver")
        .def(py::init<const std::vector<std::vector<int>> &, int, const std::vector<int> &>())
        .def("run", &RetrogradeSolver::run)
        .def("get_results", &RetrogradeSolver::get_results)
        .def("get_best_moves", &RetrogradeSolver::get_best_moves);
}