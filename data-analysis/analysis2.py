import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns


def tost_wilcoxon(a, b, margem, alpha=0.05):
    a, b = np.asarray(a), np.asarray(b)
    diff = a - b
    _, p_lower = stats.wilcoxon(diff + margem, alternative='greater')
    _, p_upper = stats.wilcoxon(diff - margem, alternative='less')
    p_tost = max(p_lower, p_upper)
    return {
        'p_lower': p_lower,
        'p_upper': p_upper,
        'p_tost': p_tost,
        'margem': margem,
        'diff_mediana': np.median(diff)
    }


def plot_wilcoxon_multi_instancia(caminhos_csv, nomes_instancias,
                                    salvar_como='grafico_wilcoxon_multi_instancia.png'):
    """
    Slide 'ingênuo': boxplot + Wilcoxon padrão para cada instância (J=5,10,15),
    com a ressalva de que p >= 0.05 NÃO prova equivalência.
    """
    fig, axes = plt.subplots(1, len(nomes_instancias), figsize=(5*len(nomes_instancias), 4.5), sharey=False)
    if len(nomes_instancias) == 1:
        axes = [axes]

    sns.set_theme(style="whitegrid")
    resultados_wilcoxon = []

    for ax, caminho, nome in zip(axes, caminhos_csv, nomes_instancias):
        df = pd.read_csv(caminho)
        stat, p_value = stats.wilcoxon(df['tempo_liao'], df['tempo_mine'])
        resultados_wilcoxon.append({'instancia': nome, 'N': len(df), 'stat': stat, 'p_valor': p_value})

        df_melted = df.melt(id_vars=['run'], value_vars=['tempo_liao', 'tempo_mine'],
                             var_name='Algoritmo', value_name='Tempo')
        df_melted['Algoritmo'] = df_melted['Algoritmo'].map({
            'tempo_liao': 'Liao e Liao (2008)', 'tempo_mine': 'Este trabalho'
        })

        sns.boxplot(x='Algoritmo', y='Tempo', data=df_melted, palette=['#e74c3c', '#2ecc71'],
                    width=0.4, boxprops=dict(alpha=0.8), showfliers=False, ax=ax)
        sns.stripplot(x='Algoritmo', y='Tempo', data=df_melted, color='black',
                      alpha=0.6, jitter=0.2, size=4, ax=ax)

        p_texto = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.4f}"
        interpretacao = "sem diferença detectável" if p_value >= 0.05 else "diferença detectada"
        ax.set_title(f"{nome} (N={len(df)})\nWilcoxon: {p_texto}\n({interpretacao})",
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Tempo de Execução (s)' if ax is axes[0] else '')

    plt.suptitle('Wilcoxon Signed-Rank por Instância\n"Ausência de evidência ≠ evidência de ausência"',
                  fontsize=11, fontweight='bold', y=1.08)
    plt.tight_layout()
    plt.savefig(salvar_como, dpi=300)
    print(f"Gráfico salvo como '{salvar_como}'")
    plt.show()

    return pd.DataFrame(resultados_wilcoxon)


def analisar_equivalencia_multi_instancia(caminhos_csv, nomes_instancias,
                                            pct_margem=0.10, alpha=0.05):
    """
    Testa se o algoritmo proposto é ESTATISTICAMENTE EQUIVALENTE ao baseline
    (Liao e Liao, 2008) em tempo de execução, através de múltiplas instâncias
    (tamanhos de job diferentes), com correção para múltiplos testes.

    pct_margem: margem de equivalência como fração da mediana do baseline
                (ex: 0.10 = 10%). Precisa ser definida A PRIORI e justificada
                no relatório — não escolha depois de ver os resultados.
    """
    resultados = []
    dfs_melted = []

    for caminho, nome in zip(caminhos_csv, nomes_instancias):
        df = pd.read_csv(caminho)

        # Checagem determinística de qualidade da solução (se aplicável)
        if 'obj_liao' in df.columns and 'obj_mine' in df.columns:
            qualidade_identica = (df['obj_liao'] == df['obj_mine']).all()
        else:
            qualidade_identica = None

        mediana_liao = df['tempo_liao'].median()
        margem = pct_margem * mediana_liao

        r = tost_wilcoxon(df['tempo_liao'], df['tempo_mine'], margem=margem, alpha=alpha)
        r['instancia'] = nome
        r['N'] = len(df)
        r['mediana_liao'] = mediana_liao
        r['mediana_mine'] = df['tempo_mine'].median()
        r['qualidade_identica'] = qualidade_identica
        resultados.append(r)

        dfm = df.melt(id_vars=[c for c in df.columns if c not in ['tempo_liao', 'tempo_mine']],
                       value_vars=['tempo_liao', 'tempo_mine'],
                       var_name='Algoritmo', value_name='Tempo')
        dfm['Algoritmo'] = dfm['Algoritmo'].map({'tempo_liao': 'Liao e Liao (2008)', 'tempo_mine': 'Este trabalho'})
        dfm['Instancia'] = nome
        dfs_melted.append(dfm)

    resultado_df = pd.DataFrame(resultados)

    # --- Correção para múltiplos testes (Holm-Bonferroni) sobre o p_tost ---
    rejeitado, p_corrigido, _, _ = multipletests(resultado_df['p_tost'], alpha=alpha, method='holm')
    resultado_df['p_tost_holm'] = p_corrigido
    resultado_df['equivalente'] = rejeitado  # rejeitar H0 do TOST = confirmar equivalência

    # --- Relatório ---
    print("\n" + "="*80)
    print("   TESTE DE EQUIVALÊNCIA (TOST) MULTI-INSTÂNCIA — Este trabalho vs. Liao (2008)")
    print("="*80)
    for _, row in resultado_df.iterrows():
        print(f"\nInstância: {row['instancia']} (N={row['N']})")
        print(f"  Mediana Liao: {row['mediana_liao']:.3f}s | Mediana Este trabalho: {row['mediana_mine']:.3f}s")
        print(f"  Margem de equivalência: ±{row['margem']:.3f}s ({pct_margem*100:.0f}% da mediana do baseline)")
        if row['qualidade_identica'] is not None:
            status_q = "✓ IDÊNTICA (determinístico)" if row['qualidade_identica'] else "✗ DIFERENTE — CUIDADO"
            print(f"  Qualidade da solução: {status_q}")
        print(f"  p-valor TOST (bruto): {row['p_tost']:.5f} | (Holm-corrigido): {row['p_tost_holm']:.5f}")
        veredito = "EQUIVALENTE" if row['equivalente'] else "NÃO EQUIVALENTE (dentro da margem)"
        print(f"  Resultado: {veredito}")
    print("="*80 + "\n")

    # --- Gráfico: diferença + IC vs. margem de equivalência, por instância ---
    fig, axes = plt.subplots(1, len(nomes_instancias), figsize=(4*len(nomes_instancias), 3.5), sharey=True)
    if len(nomes_instancias) == 1:
        axes = [axes]

    for ax, (_, row), (caminho, nome) in zip(axes, resultado_df.iterrows(), zip(caminhos_csv, nomes_instancias)):
        df = pd.read_csv(caminho)
        diff = df['tempo_liao'].values - df['tempo_mine'].values
        margem = row['margem']

        n_boot = 10000
        boot = np.array([np.median(np.random.choice(diff, len(diff), replace=True)) for _ in range(n_boot)])
        ic_lower, ic_upper = np.percentile(boot, [2.5, 97.5])

        ax.axvspan(-margem, margem, color='#2ecc71', alpha=0.15)
        ax.axvline(-margem, color='#2ecc71', linestyle='--', linewidth=1.2)
        ax.axvline(margem, color='#2ecc71', linestyle='--', linewidth=1.2)
        ax.axvline(0, color='gray', linestyle=':', linewidth=1)
        ax.errorbar(x=row['diff_mediana'], y=0.5,
                    xerr=[[row['diff_mediana']-ic_lower], [ic_upper-row['diff_mediana']]],
                    fmt='o', color='#2c3e50', ecolor='#2c3e50', elinewidth=2.5, capsize=6, markersize=8)
        ax.set_yticks([])
        ax.set_xlabel('Diferença (s)', fontsize=9)
        veredito = "Equivalente" if row['equivalente'] else "Não equiv."
        ax.set_title(f"{nome}\n{veredito} (Holm p={row['p_tost_holm']:.3f})", fontsize=9, fontweight='bold')

    plt.suptitle('Teste de Equivalência (TOST) por Instância', fontsize=11, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig('grafico_tost_multi_instancia.png', dpi=300)
    print("Gráfico salvo como 'grafico_tost_multi_instancia.png'")
    plt.show()

    return resultado_df


if __name__ == "__main__":
    caminhos = ['./results-5j.csv', './results-10j.csv', './results-15j.csv']
    nomes = ['J=5', 'J=10', 'J=15']

    # Slide "ingênuo": Wilcoxon puro, com a ressalva
    df_wilcoxon = plot_wilcoxon_multi_instancia(caminhos, nomes)

    # Slide seguinte: TOST (a prova de equivalência de verdade)
    resultado_df = analisar_equivalencia_multi_instancia(
        caminhos_csv=caminhos,
        nomes_instancias=nomes,
        pct_margem=0.10  # AJUSTE isso com justificativa própria, não arbitrariamente
    )