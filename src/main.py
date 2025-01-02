import argparse
import pandas as pd
import numpy as np
import os
from preprocess.config.config import Conf
from preprocess.clinical import ClinicalData
from preprocess.mutation import MutationData
from preprocess.mirna import miRNAData

from preprocess.config.patient_mappings import PatientMappings

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter

import matplotlib.pyplot as plt

from tqdm import tqdm

def perform_km_analysis(mutation_data: MutationData):
    df = mutation_data()

    significant_genes = []
    p_value_threshold = 0.1
    min_group_size = 20

    output_dir = "../results/km_test"
    os.makedirs(output_dir, exist_ok=True)

    for gene in tqdm(mutation_data().columns[1:-3]):
        
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
            
            plot_filename = os.path.join(output_dir, f"plots/{gene}_km_curve_paad.png")
            plt.savefig(plot_filename, dpi=300)
            plt.close()  

    significant_genes_df = pd.DataFrame(significant_genes)
    significant_genes_df.to_csv("../results/km_test/significant_genes_paad.csv")

def perform_cox_regression(mirna_data: miRNAData):
    output_dir = "../results/cox_regression"
    os.makedirs(output_dir, exist_ok=True)

    mirna_data.cox_regression(output_path = os.path.join(output_dir, 'cox_results.csv'), save_summary = True)
    mirna_data.cox_regression_results(output_dir = output_dir, save_significant_mirna = True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform Kaplan-Meier analysis on mutation data.")
    parser.add_argument("-k", "--km", action="store_true", help="Run the Kaplan-Meier analysis.")
    parser.add_argument("-c", "--cx", action="store_true", help="Run the Cox-Regression analysis.")
    args = parser.parse_args()

    clinical_data = ClinicalData(Conf.datasets['Clinical'])
    clinical_data.preprocess_data()

    patient_mappings = PatientMappings(clinical_data()['patient_id'])

    clinical_data()['pathologic_stage'].isna().sum()


    if args.km:
        print("Starting Kaplan-Meier analysis...")

        mutation_data = MutationData(Conf.datasets['miRNA'])
        mutation_data.add_stage_data(clinical_data)
        mutation_data.clean_data()

        perform_km_analysis(mutation_data)

    elif args.cx:
        print("Starting Cox-Regression analysis...")

        mirna_data = miRNAData(Conf.datasets['miRNA'])
        mirna_data.add_stage_data(clinical_data)
        mirna_data.clean_data()
        mirna_data.pca(0.85)

        perform_cox_regression(mirna_data)