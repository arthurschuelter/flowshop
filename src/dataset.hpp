#pragma once
#ifndef DATASET_HPP
#define DATASET_HPP

#include <vector>
#include <random>
#include <iostream>

// Estrutura de cada tarefa (Job)
struct Job {
    int id;
    int p1; // Tempo de processamento na Máquina 1 (p_j1)
    int p2; // Tempo de processamento na Máquina 2 (p_j2)
    int s;  // Tamanho da tarefa (s_j)
};

// Estrutura do problema/instância
struct Instance {
    int num_jobs;
    int capacity; // Capacidade das máquinas (S_m)
    std::vector<Job> jobs;
};

// 1. Instância fixa baseada na Tabela 1 do Artigo
inline Instance getExampleInstance(int num_jobs = 10) {
    Instance inst;
    inst.num_jobs = num_jobs;
    inst.capacity = 10;
    inst.jobs = {
        {1,  10, 14, 5},
        {2,   2,  9, 2},
        {3,   6, 10, 3},
        {4,  15,  1, 4},
        {5,   7, 12, 5},
        {6,   9,  4, 3},
        {7,   3,  5, 5},
        {8,  10,  8, 1},
        {9,  10,  5, 4},
        {10,  6,  9, 4}
    };
    return inst;
}

// 2. Gerador aleatório baseado nos parâmetros do artigo
// dist_type: 1 -> [1,5] (Pequenos), 2 -> [4,10] (Grandes), 3 -> [1,10] (Variado)
inline Instance generateInstance(int num_jobs, int dist_type=1, unsigned int seed = 42) {
    Instance inst;
    inst.num_jobs = num_jobs;
    inst.capacity = 10; // Capacidade padrão das máquinas
    
    int s_min = 1, s_max = 5;
    if (dist_type == 2) {
        s_min = 4; s_max = 10;
    } else if (dist_type == 3) {
        s_min = 1; s_max = 10;
    }

    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist_p(1, 15);    // Tempos de processamento U(1, 15)
    std::uniform_int_distribution<int> dist_s(s_min, s_max); // Tamanho das tarefas

    for (int j = 1; j <= num_jobs; ++j) {
        Job job;
        job.id = j;
        job.p1 = dist_p(gen);
        job.p2 = dist_p(gen);
        job.s = dist_s(gen);
        inst.jobs.push_back(job);
    }

    return inst;
}

#endif // DATASET_HPP