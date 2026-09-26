import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

def analisar_execucoes_instancia(caminho_csv=None, num_jobs=5, suffix="5j"):
    # --- 1. Carregamento / Geração dos Dados ---
    if caminho_csv:
        df = pd.read_csv(caminho_csv)
    else:
        np.random.seed(42)
        n_runs = 10
        
        tempo_liao = np.random.exponential(scale=15, size=n_runs) + 10 
        tempo_mine = np.random.exponential(scale=3, size=n_runs) + 2    
        
        df = pd.DataFrame({
            'run': [i+1 for i in range(n_runs)],
            'tempo_liao': tempo_liao,
            'tempo_mine': tempo_mine
        })

    # --- 2. Aplicação do Teste Estatístico Adequado ---
    # Usar se a execução 'i' do A usou a mesma SEED da execução 'i' do B
    stat, p_value = stats.wilcoxon(df['tempo_liao'], df['tempo_mine'])
    nome_teste = "Wilcoxon Signed-Rank (Pareado)"

    # --- 3. Métricas Descritivas (Mediana e IQR) ---
    mediana_A, q1_A, q3_A = df['tempo_liao'].median(), df['tempo_liao'].quantile(0.25), df['tempo_liao'].quantile(0.75)
    mediana_B, q1_B, q3_B = df['tempo_mine'].median(), df['tempo_mine'].quantile(0.25), df['tempo_mine'].quantile(0.75)

    # --- 4. Exibição do Relatório no Terminal ---
    print("\n" + "="*55)
    print(f"      ANÁLISE ESTATÍSTICA DA INSTÂNCIA ({nome_teste})")
    print("="*55)
    print(f"Total de Execuções por Algoritmo: N = {len(df)}")
    print(f"Liao e Liao (2008): Mediana = {mediana_A:.3f}s | IQR = [{q1_A:.3f}s - {q3_A:.3f}s]")
    print(f"Este trabalho:      Mediana = {mediana_B:.3f}s | IQR = [{q1_B:.3f}s - {q3_B:.3f}s]")
    print("-"*55)
    print(f"Estatística do Teste: {stat}")
    print(f"p-valor: {p_value:.4f}", end="")
    
    if p_value < 0.05:
        print(" => p < 0.05")
    else:
        print(" => p >= 0.05.")
    print("="*55 + "\n")

    margem_equivalencia = 0.05 * mediana_A  # exemplo: 5% da mediana do Liao
    resultado_tost = tost_wilcoxon(df['tempo_liao'], df['tempo_mine'], margem=margem_equivalencia)
    relatar_tost(resultado_tost)
    plot_tost(df, resultado_tost)

    # --- 5. Geração do Gráfico para o Slide 3 ---
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 5))

    df_melted = df.melt(id_vars=['run'], value_vars=['tempo_liao', 'tempo_mine'],
                        var_name='Algoritmo', value_name='Tempo')
    
    df_melted['Algoritmo'] = df_melted['Algoritmo'].map({
        'tempo_liao': 'Liao e Liao (2008)',
        'tempo_mine': 'Este trabalho'
    })

    # Plot dos Boxplots com pontos individuais
    ax = sns.boxplot(
        x='Algoritmo', 
        y='Tempo', 
        data=df_melted, 
        palette=['#e74c3c', '#2ecc71'], 
        width=0.4, 
        boxprops=dict(alpha=0.8),
        showfliers = False                     
        )
    
    sns.stripplot(x='Algoritmo', y='Tempo', data=df_melted, 
                  color='black', alpha=0.6, jitter=0.2, size=5)

    p_texto = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.4f}"
    plt.title(f'Distribuição dos Tempos (N={len(df)} Execuções, J={num_jobs} Tarefas)\n({nome_teste.split("(")[0].strip()}: {p_texto})', 
              fontsize=10, fontweight='bold')
    plt.ylabel('Tempo de Execução (ms)', fontsize=10)
    plt.xlabel('', fontsize=10)

    plt.tight_layout()
    plot_name = f"plot-{suffix}.png"
    plt.savefig(plot_name, dpi=300)
    print(f"Gráfico salvo como '{plot_name}'")
    plt.show()

def tost_wilcoxon(a, b, margem, alpha=0.05):
    """
    TOST (Two One-Sided Tests) para equivalência, usando Wilcoxon signed-rank
    como teste base (não-paramétrico, adequado para dados pareados sem
    normalidade garantida).
    
    Parâmetros
    ----------
    a, b : arrays pareados (mesma ordem/seed)
    margem : a margem de equivalência (delta), na mesma unidade dos dados.
             Diferenças |a - b| menores que essa margem são consideradas
             "sem relevância prática".
    alpha : nível de significância (default 0.05)
    
    Retorna
    -------
    dict com resultados dos dois testes unilaterais e a conclusão
    """
    a = np.asarray(a)
    b = np.asarray(b)
    diff = a - b

    # H1_lower: diferença média/mediana > -margem  (a não é muito menor que b)
    # H1_upper: diferença média/mediana < +margem  (a não é muito maior que b)
    
    # Teste 1: (a - b) + margem > 0  ->  desloca e testa se é > 0
    stat_lower, p_lower = stats.wilcoxon(diff + margem, alternative='greater')
    
    # Teste 2: (a - b) - margem < 0  ->  desloca e testa se é < 0
    stat_upper, p_upper = stats.wilcoxon(diff - margem, alternative='less')

    p_tost = max(p_lower, p_upper)  # o mais conservador dos dois define o resultado
    equivalente = p_tost < alpha

    return {
        'p_lower': p_lower,
        'p_upper': p_upper,
        'p_tost': p_tost,
        'margem': margem,
        'equivalente': equivalente,
        'diff_mediana': np.median(diff)
    }

def relatar_tost(resultado, nome_A="Liao e Liao (2008)", nome_B="Este trabalho"):
    print("\n" + "="*55)
    print("      TESTE DE EQUIVALÊNCIA (TOST)")
    print("="*55)
    print(f"Margem de equivalência: ±{resultado['margem']}")
    print(f"Diferença mediana observada (A - B): {resultado['diff_mediana']:.4f}")
    print(f"p-valor (limite inferior): {resultado['p_lower']:.4f}")
    print(f"p-valor (limite superior): {resultado['p_upper']:.4f}")
    print(f"p-valor TOST (max dos dois): {resultado['p_tost']:.4f}")
    print("-"*55)
    if resultado['equivalente']:
        print(f"=> Equivalência estatística CONFIRMADA (p < 0.05)")
        print(f"   Os algoritmos são equivalentes dentro da margem de ±{resultado['margem']}.")
    else:
        print(f"=> Equivalência NÃO confirmada (p >= 0.05)")
        print(f"   Não há evidência suficiente de equivalência dentro dessa margem.")
    print("="*55 + "\n")

def calcular_ic_diferenca(diff, alpha=0.05):
    """
    Calcula o intervalo de confiança da mediana das diferenças via bootstrap
    (mais robusto que assumir normalidade, consistente com a filosofia
    não-paramétrica do Wilcoxon/TOST).
    """
    n_boot = 10000
    boot_medianas = np.array([
        np.median(np.random.choice(diff, size=len(diff), replace=True))
        for _ in range(n_boot)
    ])
    ic_lower = np.percentile(boot_medianas, 100 * alpha / 2)
    ic_upper = np.percentile(boot_medianas, 100 * (1 - alpha / 2))
    return ic_lower, ic_upper

def plot_tost(df, resultado_tost, nome_A="Liao e Liao (2008)", nome_B="Este trabalho",
              salvar_como='grafico_tost.png'):
    """
    Gráfico clássico de TOST: mostra a diferença observada (A - B) com seu
    intervalo de confiança, sobreposto à faixa de equivalência [-margem, +margem].
    """
    diff = df['tempo_liao'].values - df['tempo_mine'].values
    margem = resultado_tost['margem']
    diff_mediana = resultado_tost['diff_mediana']

    ic_lower, ic_upper = calcular_ic_diferenca(diff)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 3.5))

    # Faixa de equivalência (zona "aceitável")
    ax.axvspan(-margem, margem, color='#2ecc71', alpha=0.15, label='Zona de equivalência')
    ax.axvline(-margem, color='#2ecc71', linestyle='--', linewidth=1.2)
    ax.axvline(margem, color='#2ecc71', linestyle='--', linewidth=1.2)

    # Linha de diferença zero (referência)
    ax.axvline(0, color='gray', linestyle=':', linewidth=1)

    # Diferença observada + IC
    ax.errorbar(
        x=diff_mediana, y=0.5,
        xerr=[[diff_mediana - ic_lower], [ic_upper - diff_mediana]],
        fmt='o', color='#2c3e50', ecolor='#2c3e50',
        elinewidth=2.5, capsize=6, markersize=9,
        label='Diferença mediana (A - B) + IC 95%'
    )

    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.set_xlabel(f'Diferença de tempo: {nome_A} - {nome_B} (s)', fontsize=10)

    dentro = (ic_lower > -margem) and (ic_upper < margem)
    conclusao = "Equivalência confirmada" if dentro else "Equivalência NÃO confirmada"
    ax.set_title(f'Teste de Equivalência (TOST)\nMargem: ±{margem:.3f}s  |  {conclusao}',
                  fontsize=10, fontweight='bold')

    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(salvar_como, dpi=300)
    print(f"Gráfico TOST salvo como '{salvar_como}'")
    plt.show()

if __name__ == "__main__":
    suffix = '5j'
    caminho_csv = f"./results-{suffix}.csv";
    num_jobs = 10
    analisar_execucoes_instancia(
        caminho_csv=caminho_csv, 
        num_jobs=num_jobs,
        suffix=suffix
    )