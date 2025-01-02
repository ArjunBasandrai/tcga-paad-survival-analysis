import pandas as pd

import os
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
from tqdm import tqdm

from preprocess.clinical import ClinicalData

class MutationData:
    def __init__(self, path: str):
        self.df: pd.DataFrame = self.load_and_transpose_df(path)
        self.label_mappings: dict = {}
    
    def load_and_transpose_df(self, dataset: str) -> pd.DataFrame:
        df = pd.read_csv(dataset).T
        df.columns = df.iloc[0]
        df = df[1:]
        df = df.reset_index()
        df = df.rename(columns={'index': 'patient_id'})
        df.columns.name = None
        return df

    def merge_data(self, clinical_data: ClinicalData) -> None:
        self.df = self.df.merge(
            clinical_data()[['patient_id', 'status', 'overall_survival']],
            on='patient_id',
            how='left'
        )

    def select_common_patients(self) -> None:
        self.df.dropna(subset=['status', 'overall_survival'], inplace=True)

    def clean_data(self) -> None:
        for col in self.df.columns[1:]:
            self.df[col] = self.df[col].astype(int)

    def add_stage_data(self, clinical_data: ClinicalData) -> None:
        self.merge_data(clinical_data)
        self.select_common_patients()
    
    def kaplan_meier_analysis(self, output_dir: str) -> None:
        significant_genes = []
        p_value_threshold = 0.1
        min_group_size = 20

        for gene in tqdm(self.df.columns[1:-3]):
            
            mutated = self.df[self.df[gene] == 1]
            non_mutated = self.df[self.df[gene] == 0]

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
        
    def __call__(self) -> pd.DataFrame:
        return self.df