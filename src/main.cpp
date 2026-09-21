#include "gurobi_c++.h"
#include <string>
#include <vector>
#include<iostream>
#include <random>

void print_results(
        GRBModel& model,
        int num_jobs, 
        int num_machines, 
        int num_batches,
        std::vector<std::vector<GRBVar>>& X,
        std::vector<std::vector<GRBVar>>& P,
        std::vector<std::vector<GRBVar>>& C);


int main(int argc, char* argv[]) {
    // ---------------------------------------------------------------
    // 1) Problem data
    //    p[i][j] = processing time of job i on machine j
    //    B[j]    = batch capacity (max jobs per batch) on machine j
    // ---------------------------------------------------------------

    int num_jobs = 5;
    int num_machines = 2;                               // Machine 1 and Machine 2
    // int num_batches = (num_jobs * num_machines) + 1;    // Maximum number of possible batches (|B|)
    int num_batches = 5;    // Maximum number of possible batches (|B|)
    
    double S_min = 10.0;    // Batch capacity parameter S_min
    
    // s[j]: Size of job j
    std::vector<double> s = {3.0, 4.0, 2.0, 5.0, 3.0};
    
    // p[j][m]: Processing time of job j on machine m
    std::vector<std::vector<double>> p = {
        {2.0, 3.0}, // Job 0
        {4.0, 2.0}, // Job 1
        {1.0, 5.0}, // Job 2
        {3.0, 3.0}, // Job 3
        {2.0, 1.0}  // Job 4
    };

    try {
        // --- Gurobi Environment and Model Setup ---
        GRBEnv env = GRBEnv(true);
        env.set(GRB_IntParam_OutputFlag, 1);
        env.start();

        GRBModel model = GRBModel(env);
        model.set(GRB_StringAttr_ModelName, "FlowShop_Batch_Scheduling");

        // --- Decision Variables ---

        // X[j][b]: Eq. (13') - Binary variable, 1 if job j is assigned to batch b
        std::vector<std::vector<GRBVar>> X(num_jobs, std::vector<GRBVar>(num_batches));
        for (int j = 0; j < num_jobs; ++j) {
            for (int b = 0; b < num_batches; ++b) {
                X[j][b] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, 
                                       "X_" + std::to_string(j) + "_" + std::to_string(b));
            }
        }

        // P[b][m]: Eq. (14') - Processing time of batch b on machine m
        std::vector<std::vector<GRBVar>> P(num_batches, std::vector<GRBVar>(num_machines));
        for (int b = 0; b < num_batches; ++b) {
            for (int m = 0; m < num_machines; ++m) {
                P[b][m] = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, 
                                       "P_" + std::to_string(b) + "_" + std::to_string(m));
            }
        }

        // C[b][m]: Eq. (14') - Completion time of batch b on machine m
        std::vector<std::vector<GRBVar>> C(num_batches, std::vector<GRBVar>(num_machines));
        for (int b = 0; b < num_batches; ++b) {
            for (int m = 0; m < num_machines; ++m) {
                C[b][m] = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, 
                                       "C_" + std::to_string(b) + "_" + std::to_string(m));
            }
        }

        // Cmax: Eq. (14') - Makespan
        GRBVar Cmax = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "Cmax");

        // --- Objective Function (1') ---
        model.setObjective(GRBLinExpr(Cmax), GRB_MINIMIZE);

        // --- Constraints ---

        // Eq. (2'): Each job assigned to exactly one batch
        for (int j = 0; j < num_jobs; ++j) {
            GRBLinExpr sum_X = 0;
            for (int b = 0; b < num_batches; ++b) {
                sum_X += X[j][b];
            }
            model.addConstr(sum_X == 1, "JobAssignment_" + std::to_string(j));
        }

        // Eq. (3'): Batch size capacity constraint
        for (int b = 0; b < num_batches; ++b) {
            GRBLinExpr size_sum = 0;
            for (int j = 0; j < num_jobs; ++j) {
                size_sum += s[j] * X[j][b];
            }
            model.addConstr(size_sum <= S_min, "BatchCapacity_" + std::to_string(b));
        }

        // Eq. (4'): Batch processing time determination
        for (int j = 0; j < num_jobs; ++j) {
            for (int b = 0; b < num_batches; ++b) {
                for (int m = 0; m < num_machines; ++m) {
                    model.addConstr(P[b][m] >= p[j][m] * X[j][b], 
                                     "BatchProcTime_" + std::to_string(j) + "_" + 
                                     std::to_string(b) + "_" + std::to_string(m));
                }
            }
        }

        // Eq. (5'): Completion time of batch b on Machine 1 (m = 0)
        for (int b = 0; b < num_batches; ++b) {
            GRBLinExpr sum_P = 0;
            for (int k = 0; k <= b; ++k) {
                sum_P += P[k][0];
            }
            model.addConstr(C[b][0] == sum_P, "Completion_M1_" + std::to_string(b));
        }

        // Eq. (6'): Completion time of first batch on Machine 2 (b = 0, m = 1)
        model.addConstr(C[0][1] == C[0][0] + P[0][1], "Completion_M2_FirstBatch");

        // Eq. (10'): Sequential precedence of batches on Machine 2
        for (int b = 1; b < num_batches; ++b) {
            model.addConstr(C[b][1] >= C[b - 1][1] + P[b][1], 
                            "Completion_M2_Seq_" + std::to_string(b));
        }

        // Eq. (11'): Machine 2 can only process batch b after it completes on Machine 1
        for (int b = 0; b < num_batches; ++b) {
            model.addConstr(C[b][1] - C[b][0] >= P[b][1], 
                            "Completion_M2_M1_" + std::to_string(b));
        }

        // Eq. (12'): Makespan definition
        model.addConstr(Cmax >= C[num_batches - 1][1], "Makespan");

        // --- Solve Optimization Model ---
        model.optimize();

        // --- Print Results ---
        if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL) print_results(model, num_jobs, num_machines, num_batches, X, P, C);


    } catch (GRBException& e) {
        std::cerr << "Gurobi error code = " << e.getErrorCode() << "\n";
        std::cerr << e.getMessage() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception during optimization.\n";
        return 1;
    }

    return 0;
}

void print_results(
        GRBModel& model,
        int num_jobs, 
        int num_machines, 
        int num_batches,
        std::vector<std::vector<GRBVar>>& X,
        std::vector<std::vector<GRBVar>>& P,
        std::vector<std::vector<GRBVar>>& C
    ) {
    std::cout << "\n=============================================";
    std::cout << "\nOptimal Makespan (Cmax): " << model.get(GRB_DoubleAttr_ObjVal);
    std::cout << "\n=============================================\n";

    for (int m = 0; m < num_machines; ++m) {
        std::cout << "\n--- MACHINE " << (m + 1) << " SCHEDULE ---\n";
        for (int b = 0; b < num_batches; ++b) {
            double p_bm = P[b][m].get(GRB_DoubleAttr_X);
            double c_bm = C[b][m].get(GRB_DoubleAttr_X);
            double start_time = c_bm - p_bm;

            // Only print batches that have non-zero processing time
            if (p_bm > 1e-5) {
                std::cout << "  Batch " << (b + 1) 
                        << " [Time " << start_time << " -> " << c_bm << "]"
                        << " (Duration: " << p_bm << "): Jobs { ";
                for (int j = 0; j < num_jobs; ++j) {
                    if (X[j][b].get(GRB_DoubleAttr_X) > 0.5) {
                        std::cout << j << " ";
                    }
                }
                std::cout << "}\n";
            }
        }
    }
}