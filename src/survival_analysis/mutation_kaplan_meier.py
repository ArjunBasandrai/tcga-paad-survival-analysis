import os
import pandas as pd

import matplotlib.pyplot as plt
from tqdm import tqdm
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

def kaplan_meier_analysis(df: pd.DataFrame, output_dir: str) -> None:
    significant_genes = []
    p_value_threshold = 0.1
    min_group_size = 20

    for gene in tqdm(df.columns[1:-3]):
        
        mutated = df[df[gene] == 1]
        non_mutated = df[df[gene] == 0]

        if len(mutated) < min_group_size or len(non_mutated) < min_group_size:
            continue

        results_logrank = logrank_test(
            mutated["overall_survival"], non_mutated["overall_survival"],
            event_observed_A=mutated["status"], event_observed_B=non_mutated["status"]
        )

        if results_logrank.p_value < p_value_threshold:
            significant_genes.append({
                "gene": gene,
                "mutated_count": len(mutated),
                "non_mutated_count": len(non_mutated),
                "mutated_mean_survival (days)": round(mutated['overall_survival'].mean(), 2),
                "non_mutated_mean_survival (days)": round(non_mutated['overall_survival'].mean(), 2),
                "p_value": results_logrank.p_value,
            })
        
            kmf = KaplanMeierFitter()
            
            plt.figure(figsize=(10, 6))
            kmf.fit(mutated["overall_survival"], event_observed=mutated["status"], label="Mutated")
            kmf.plot_survival_function()
            
            kmf.fit(non_mutated["overall_survival"], event_observed=non_mutated["status"], label="Non-Mutated")
            kmf.plot_survival_function()
            
            plt.title(f"Kaplan-Meier Survival Curve for {gene} Mutation")
            plt.xlabel("Time (days)")
            plt.ylabel("Survival Probability")
            plt.legend()
            plt.grid(True)

            _output_dir = os.path.join(output_dir, "plots")
            os.makedirs(_output_dir, exist_ok=True)
                            
            plot_filename = os.path.join(_output_dir, f"{gene}_km_curve_paad.png")
            plt.savefig(plot_filename, dpi=300)
            plt.close()  

    significant_genes_df = pd.DataFrame(significant_genes)
    significant_genes_df.to_csv(os.path.join(output_dir, "significant_genes_paad.csv"))