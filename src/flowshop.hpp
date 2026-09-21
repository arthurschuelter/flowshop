#pragma once


#include "gurobi_c++.h"
#include <iostream>
#include <chrono>

struct ModelResults {
    GRBModel* model;
    std::string model_name;
    std::vector<std::chrono::duration<double>> elapsed;
};

class Flowshop {
public:
    Flowshop(GRBEnv& env, int num_jobs, int num_machines, int num_batches, double S_min, std::vector<double> s, std::vector<std::vector<double>> p, std::string model_name);
    ~Flowshop() = default;

    void addConstraints();
    void addObjective();
    void addDecisionVariables();

    ModelResults* optimize();

    std::vector<std::vector<GRBVar>> X;
    std::vector<std::vector<GRBVar>> P;
    std::vector<std::vector<GRBVar>> C;

private:
    GRBModel* model;

    int num_jobs;
    int num_machines;
    int num_batches;
    double S_min;
    std::string model_name;

    std::vector<double> s;
    std::vector<std::vector<double>> p; 
    GRBVar Cmax;

};