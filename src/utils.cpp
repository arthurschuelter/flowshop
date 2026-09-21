#include "utils.hpp"

void print_results(ModelResults& results) {
    
    printf("\n=============================================\n");
    printf("Model: %s\n", results.model_name.c_str());
    printf("Optimal Makespan (Cmax): %.2f\n", results.model->get(GRB_DoubleAttr_ObjVal));
    // printf("Execution Time: %.2f ms\n", duration_ms.count());
    printf("Execution Time: \n");
    for (int i = 0; i < results.elapsed.size(); i++) {
        std::chrono::duration<double, std::milli> duration_ms = results.elapsed[i];
        printf("  Run %d: %.2f ms\n", i, duration_ms.count());
    }
    printf("=============================================\n");
}

void print_solution(
        ModelResults& results,
        int num_jobs, 
        int num_machines, 
        int num_batches,
        std::vector<std::vector<GRBVar>>& X,
        std::vector<std::vector<GRBVar>>& P,
        std::vector<std::vector<GRBVar>>& C
    ) {

    print_results(results);

    for (int m = 0; m < num_machines; ++m) {
        std::cout << "\n--- MACHINE " << (m + 1) << " SCHEDULE ---\n";
        for (int b = 0; b < num_batches; ++b) {
            double p_bm = P[b][m].get(GRB_DoubleAttr_X);
            double c_bm = C[b][m].get(GRB_DoubleAttr_X);
            double start_time = c_bm - p_bm;

            // Only print batches that have non-zero processing time
            if (p_bm > 1e-5) {
                printf("  Batch %d [Time = %2.0f -> %2.0f] (Duration = %2.0f): Jobs { ", 
                        b + 1, start_time, c_bm, p_bm);
                for (int j = 0; j < num_jobs; ++j) {
                    if (X[j][b].get(GRB_DoubleAttr_X) > 0.5) {
                        std::cout << j << " ";
                    }
                }
                std::cout << "}\n";
            }
        }
    }
    std::cout << "\n=============================================\n";
}