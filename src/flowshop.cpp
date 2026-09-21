#include "flowshop.hpp"

Flowshop::Flowshop(
        GRBEnv& env, 
        int num_jobs, 
        int num_machines, 
        int num_batches, 
        double S_min, 
        std::vector<double> s, 
        std::vector<std::vector<double>> p, 
        std::string model_name
    ) : num_jobs(num_jobs), 
        num_machines(num_machines), 
        num_batches(num_batches), 
        S_min(S_min), 
        s(s), 
        p(p), 
        model_name(model_name)
        {
    this->model = new GRBModel(env);
    this->model->set(GRB_StringAttr_ModelName, model_name);
}

void Flowshop::addDecisionVariables() {
    // X[j][b]: Eq. (13') - Binary variable, 1 if job j is assigned to batch b
    this->X = std::vector<std::vector<GRBVar>>(num_jobs, std::vector<GRBVar>(num_batches));
    for (int j = 0; j < num_jobs; ++j) {
        for (int b = 0; b < num_batches; ++b) {
            this->X[j][b] = this->model->addVar(0.0, 1.0, 0.0, GRB_BINARY, 
                                                "X_" + std::to_string(j) + "_" + std::to_string(b));
        }
    }

    // P[b][m]: Eq. (14') - Processing time of batch b on machine m
    this->P = std::vector<std::vector<GRBVar>>(num_batches, std::vector<GRBVar>(num_machines));
    for (int b = 0; b < num_batches; ++b) {
        for (int m = 0; m < num_machines; ++m) {
            this->P[b][m] = this->model->addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, 
                                                "P_" + std::to_string(b) + "_" + std::to_string(m));
        }
    }

    // C[b][m]: Eq. (14') - Completion time of batch b on machine m
    this->C = std::vector<std::vector<GRBVar>>(num_batches, std::vector<GRBVar>(num_machines));
    for (int b = 0; b < num_batches; ++b) {
        for (int m = 0; m < num_machines; ++m) {
            this->C[b][m] = this->model->addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, 
                                                "C_" + std::to_string(b) + "_" + std::to_string(m));
        }
    }

    // Cmax: Eq. (14') - Makespan
    this->Cmax = this->model->addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "Cmax");
}

void Flowshop::addObjective() {
    // --- Objective Function (1') ---
    this->model->setObjective(GRBLinExpr(this->Cmax), GRB_MINIMIZE);
}

void Flowshop::addConstraints() {
    // Eq. (2'): Each job assigned to exactly one batch
    for (int j = 0; j < num_jobs; ++j) {
        GRBLinExpr sum_X = 0;
        for (int b = 0; b < num_batches; ++b) {
            sum_X += this->X[j][b];
        }
        this->model->addConstr(sum_X == 1, "JobAssignment_" + std::to_string(j));
    }

    // Eq. (3'): Batch size capacity constraint
    for (int b = 0; b < num_batches; ++b) {
        GRBLinExpr size_sum = 0;
        for (int j = 0; j < num_jobs; ++j) {
            size_sum += this->s[j] * this->X[j][b];
        }
        this->model->addConstr(size_sum <= this->S_min, "BatchCapacity_" + std::to_string(b));
    }

    // Eq. (4'): Batch processing time determination
    for (int j = 0; j < num_jobs; ++j) {
        for (int b = 0; b < num_batches; ++b) {
            for (int m = 0; m < num_machines; ++m) {
                this->model->addConstr(this->P[b][m] >= this->p[j][m] * this->X[j][b], 
                                        "BatchProcTime_" + std::to_string(j) + "_" + 
                                        std::to_string(b) + "_" + std::to_string(m));
            }
        }
    }

    // --- Machine 1 (m = 0): Completion time is the cumulative sum of processing times ---
    for (int b = 0; b < num_batches; ++b) {
        GRBLinExpr sum_P = 0;
        for (int k = 0; k <= b; ++k) {
            sum_P += this->P[k][0];
        }
        this->model->addConstr(this->C[b][0] == sum_P, "Completion_M0_Batch_" + std::to_string(b));
    }

    // --- Subsequent Machines (m >= 1) ---
    for (int m = 1; m < num_machines; ++m) {
        
        // (Generalizes Eq. 6' and Eq. 11')
        for (int b = 0; b < num_batches; ++b) {
            this->model->addConstr(
                this->C[b][m] >= this->C[b][m - 1] + this->P[b][m],
                "FlowPrecedence_M" + std::to_string(m) + "_Batch_" + std::to_string(b)
            );
        }

        // (Generalizes Eq. 10')
        for (int b = 1; b < num_batches; ++b) {
            this->model->addConstr(
                this->C[b][m] >= this->C[b - 1][m] + this->P[b][m],
                "MachinePrecedence_M" + std::to_string(m) + "_Batch_" + std::to_string(b)
            );
        }
    }

    // Eq. (12'): Makespan definition
    this->model->addConstr(Cmax >= this->C[num_batches - 1][num_machines - 1], "Makespan");
}


void Flowshop::addConstraints_M2() {
    // Eq. (2'): Each job assigned to exactly one batch
    for (int j = 0; j < num_jobs; ++j) {
        GRBLinExpr sum_X = 0;
        for (int b = 0; b < num_batches; ++b) {
            sum_X += this->X[j][b];
        }
        this->model->addConstr(sum_X == 1, "JobAssignment_" + std::to_string(j));
    }

    // Eq. (3'): Batch size capacity constraint
    for (int b = 0; b < num_batches; ++b) {
        GRBLinExpr size_sum = 0;
        for (int j = 0; j < num_jobs; ++j) {
            size_sum += this->s[j] * this->X[j][b];
        }
        this->model->addConstr(size_sum <= this->S_min, "BatchCapacity_" + std::to_string(b));
    }

    // Eq. (4'): Batch processing time determination
    for (int j = 0; j < num_jobs; ++j) {
        for (int b = 0; b < num_batches; ++b) {
            for (int m = 0; m < num_machines; ++m) {
                this->model->addConstr(this->P[b][m] >= this->p[j][m] * this->X[j][b], 
                                        "BatchProcTime_" + std::to_string(j) + "_" + 
                                        std::to_string(b) + "_" + std::to_string(m));
            }
        }
    }

    // Eq. (5'): Completion time of batch b on Machine 1 (m = 0)
    for (int b = 0; b < num_batches; ++b) {
        GRBLinExpr sum_P = 0;
        for (int k = 0; k <= b; ++k) {
            sum_P += this->P[k][0];
        }
        this->model->addConstr(this->C[b][0] == sum_P, "Completion_M1_" + std::to_string(b));
    }

    // Eq. (6'): Completion time of first batch on Machine 2 (b = 0, m = 1)
    this->model->addConstr(this->C[0][1] == this->C[0][0] + this->P[0][1], "Completion_M2_FirstBatch");

    // Eq. (10'): Sequential precedence of batches on Machine 2
    for (int b = 1; b < num_batches; ++b) {
        this->model->addConstr(this->C[b][1] >= this->C[b - 1][1] + this->P[b][1], 
                        "Completion_M2_Seq_" + std::to_string(b));
    }

    // Eq. (11'): Machine 2 can only process batch b after it completes on Machine 1
    for (int b = 0; b < num_batches; ++b) {
        this->model->addConstr(this->C[b][1] - this->C[b][0] >= this->P[b][1], 
                        "Completion_M2_M1_" + std::to_string(b));
    }

    // Eq. (12'): Makespan definition
    this->model->addConstr(this->Cmax >= this->C[num_batches - 1][1], "Makespan");
}

ModelResults* Flowshop::optimize() {
    auto start = std::chrono::high_resolution_clock::now();

    addDecisionVariables();
    addObjective();
    addConstraints();
    this->model->optimize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    ModelResults* results = new ModelResults();
    results->model = this->model;
    results->model_name = this->model_name;
    results->elapsed.push_back(elapsed);
    return results;
}

ModelResults* Flowshop::optimize_Liao() {
    auto start = std::chrono::high_resolution_clock::now();

    addDecisionVariables();
    addObjective();
    addConstraints_M2();
    this->model->optimize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    ModelResults* results = new ModelResults();
    results->model = this->model;
    results->model_name = this->model_name;
    results->elapsed.push_back(elapsed);
    return results;
}