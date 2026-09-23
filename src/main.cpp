#include "utils.hpp"
#include "flowshop.hpp"
#include "dataset.hpp"

int main(int argc, char* argv[]) {
    int num_jobs = 5;
    int num_machines = 2;
    int num_batches = 5;

    int iterations = 30;
    
    Instance inst = generateInstance(num_jobs);
    
    for (const auto& job : inst.jobs) {
        printf("Job %2d | p1: %2d | p2: %2d | s: %2d\n", job.id, job.p1, job.p2, job.s);
    }
        
    double S_min = 10.0;
    
    // s[j]: Size of job j
    std::vector<double> s;
    std::transform(inst.jobs.begin(), inst.jobs.end(), 
                    std::back_inserter(s), 
                    [](const Job& job) { return static_cast<double>(job.s); });
    
    // p[j][m]: Processing time of job j on machine m
    std::vector<std::vector<double>> p;
    std::transform(inst.jobs.begin(), inst.jobs.end(), 
                    std::back_inserter(p), 
                    [](const Job& job) { return std::vector<double>{static_cast<double>(job.p1), static_cast<double>(job.p2)}; });

    ModelResults results_liao_final = {
        nullptr,
        "Liao 2008",
        std::vector<std::chrono::duration<double>>()
    };

    ModelResults results_mine_final = {
        nullptr,
        "Mine 2026",
        std::vector<std::chrono::duration<double>>()
    };
    
    try {

        GRBEnv env_ = GRBEnv(true);
        env_.set(GRB_IntParam_OutputFlag, 0);
        env_.start();
        
        Flowshop flowshop = Flowshop(env_, num_jobs, num_machines, num_batches, S_min, s, p, "Liao 2008");
        ModelResults* results = flowshop.optimize_Liao();


        for (int it = 0; it < iterations; it++) {
            // Liao, Liao 2008
            GRBEnv env = GRBEnv(true);
            env.set(GRB_IntParam_OutputFlag, 0);
            env.start();
            
            Flowshop flowshop_liao = Flowshop(env, num_jobs, num_machines, num_batches, S_min, s, p, "Liao 2008");
            ModelResults* results_liao = flowshop_liao.optimize_Liao();
            results_liao_final.elapsed.push_back(results_liao->elapsed.back());
            if (results_liao->model->get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
                results_liao_final.model = results_liao->model;
            }
                
            // Mine 2026
            GRBEnv env2 = GRBEnv(true);
            env2.set(GRB_IntParam_OutputFlag, 0);
            env2.start();
            Flowshop flowshop_mine = Flowshop(env2, num_jobs, num_machines, num_batches, S_min, s, p, "Mine 2026");
            ModelResults* results_mine = flowshop_mine.optimize();
            results_mine_final.elapsed.push_back(results_mine->elapsed.back());
                
            if (results_mine->model->get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
                results_mine_final.model = results_mine->model;
            }
                
        }

        // print_results(results_liao_final);
        // print_results(results_mine_final);

        printf("run,tempo_liao,tempo_mine\n");
        for (int i = 0; i < results_liao_final.elapsed.size(); i++) {
            std::chrono::duration<double, std::milli> duration_liao = results_liao_final.elapsed[i];
            std::chrono::duration<double, std::milli> duration_mine = results_mine_final.elapsed[i];
            printf("%d,%.3f,%.3f\n", i+1, duration_liao.count(), duration_mine.count());
        }

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
