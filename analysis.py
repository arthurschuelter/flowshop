import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

def analisar_execucoes_instancia(caminho_csv=None, pareado=False):
    # --- 1. Carregamento / Geração dos Dados ---
    caminho_csv = "./results.csv";
    if caminho_csv:
        df = pd.read_csv(caminho_csv)
    else:
        # Dados sintéticos de exemplo (30 execuções para a mesma instância)
        np.random.seed(42)
        n_runs = 30
        
        # Algoritmo A: Modelo Padrão
        tempo_liao = np.random.exponential(scale=15, size=n_runs) + 10 
        # Algoritmo B: Com Quebra de Simetria (mais rápido)
        tempo_mine = np.random.exponential(scale=3, size=n_runs) + 2    
        
        df = pd.DataFrame({
            'run': [i+1 for i in range(n_runs)],
            'tempo_liao': tempo_liao,
            'tempo_mine': tempo_mine
        })

    # --- 2. Aplicação do Teste Estatístico Adequado ---
    if pareado:
        # Usar se a execução 'i' do A usou a mesma SEED da execução 'i' do B
        stat, p_value = stats.wilcoxon(df['tempo_liao'], df['tempo_mine'])
        nome_teste = "Wilcoxon Signed-Rank (Pareado)"
    else:
        # Usar para execuções independentes (Padrão)
        stat, p_value = stats.mannwhitneyu(df['tempo_liao'], df['tempo_mine'], alternative='two-sided')
        nome_teste = "Mann-Whitney U (Amostras Independentes)"

    # --- 3. Métricas Descritivas (Mediana e IQR) ---
    mediana_A, q1_A, q3_A = df['tempo_liao'].median(), df['tempo_liao'].quantile(0.25), df['tempo_liao'].quantile(0.75)
    mediana_B, q1_B, q3_B = df['tempo_mine'].median(), df['tempo_mine'].quantile(0.25), df['tempo_mine'].quantile(0.75)

    # --- 4. Exibição do Relatório no Terminal ---
    print("\n" + "="*55)
    print(f"      ANÁLISE ESTATÍSTICA DA INSTÂNCIA ({nome_teste})")
    print("="*55)
    print(f"Total de Execuções por Algoritmo: N = {len(df)}")
    print(f"Algoritmo A (Sem Quebra): Mediana = {mediana_A:.3f}s | IQR = [{q1_A:.3f}s - {q3_A:.3f}s]")
    print(f"Algoritmo B (Com Quebra): Mediana = {mediana_B:.3f}s | IQR = [{q1_B:.3f}s - {q3_B:.3f}s]")
    print("-"*55)
    print(f"Estatística do Teste: {stat}")
    print(f"p-valor: {p_value:.5e}")
    
    if p_value < 0.05:
        print("\n CONCLUSÃO: A diferença no tempo de execução é estatisticamente significativa (p < 0.05).")
    else:
        print("\n CONCLUSÃO: NÃO há evidência de diferença significativa (p >= 0.05).")
    print("="*55 + "\n")

    # --- 5. Geração do Gráfico para o Slide 3 ---
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 5))

    df_melted = df.melt(id_vars=['run'], value_vars=['tempo_liao', 'tempo_mine'],
                        var_name='Algoritmo', value_name='Tempo')
    
    df_melted['Algoritmo'] = df_melted['Algoritmo'].map({
        'tempo_liao': 'Algoritmo A\n(Sem Quebra)',
        'tempo_mine': 'Algoritmo B\n(Com Quebra)'
    })

    # Plot dos Boxplots com pontos individuais
    ax = sns.boxplot(x='Algoritmo', y='Tempo', data=df_melted, 
                     palette=['#e74c3c', '#2ecc71'], width=0.4, boxprops=dict(alpha=0.8))
    sns.stripplot(x='Algoritmo', y='Tempo', data=df_melted, 
                  color='black', alpha=0.6, jitter=0.2, size=5)

    p_texto = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.4f}"
    plt.title(f'Distribuição dos Tempos em N={len(df)} Execuções\n({nome_teste.split("(")[0].strip()}: {p_texto})', 
              fontsize=10, fontweight='bold')
    plt.ylabel('Tempo de Execução (segundos)', fontsize=10)
    plt.xlabel('', fontsize=10)

    plt.tight_layout()
    plt.savefig('grafico_slide3.png', dpi=300)
    print("Gráfico salvo como 'grafico_slide3.png'")
    plt.show()

if __name__ == "__main__":
    # mude pareado=True se tiver pareado as seeds nas execuções
    analisar_execucoes_instancia(pareado=False)