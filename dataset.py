import random

def generate_flowshop_dataset(num_jobs=10, dist_type=1, seed=None):
    if seed is not None:
        random.seed(seed)
        
    capacity = 10
    
    # Seleção do intervalo de tamanhos de acordo com a distribuição do artigo
    if dist_type == 1:
        s_min, s_max = 1, 5
    elif dist_type == 2:
        s_min, s_max = 4, 10
    elif dist_type == 3:
        s_min, s_max = 1, 10
    else:
        raise ValueError("dist_type deve ser 1, 2 ou 3")
        
    jobs = []
    for j in range(1, num_jobs + 1):
        p_j1 = random.randint(1, 15) # Tempo máq. 1
        p_j2 = random.randint(1, 15) # Tempo máq. 2
        s_j = random.randint(s_min, s_max) # Tamanho da tarefa
        
        jobs.append({
            "job": j,
            "p_j1": p_j1,
            "p_j2": p_j2,
            "s_j": s_j
        })
        
    return capacity, jobs

# Exemplo de utilização: 10 tarefas com Distribuição I
capacity, dataset = generate_flowshop_dataset(num_jobs=10, dist_type=1, seed=42)

print(f"Capacidade da Máquina: {capacity}\n")
print(f"{'Job':<6}{'p_j1':<8}{'p_j2':<8}{'s_j':<6}")
print("-" * 28)
for job in dataset:
    print(f"{job['job']:<6}{job['p_j1']:<8}{job['p_j2']:<8}{job['s_j']:<6}")