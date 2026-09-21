#pragma once

#include "gurobi_c++.h"
#include "flowshop.hpp"

#include <string>
#include <vector>
#include <iostream>
#include <random>
#include <algorithm>

void print_results(ModelResults& results);

void print_solution(
        ModelResults& results,
        int num_jobs, 
        int num_machines, 
        int num_batches,
        std::vector<std::vector<GRBVar>>& X,
        std::vector<std::vector<GRBVar>>& P,
        std::vector<std::vector<GRBVar>>& C
    );
