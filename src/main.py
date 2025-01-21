import argparse
import os
import numpy as np
import random

from config.config import Conf
from config.patient_mappings import PatientMappings
from data_classes.clinical import ClinicalData
from data_classes.mutation import MutationData
from data_classes.mirna import miRNAData
from data_classes.rna import RNAData

from survival_analysis.mutation_kaplan_meier import kaplan_meier_analysis
from survival_analysis.mirna_cox_regression import cox_regression, cox_regression_results
from survival_analysis.rna_cox_regression import rna_cox_regression

np.random.seed(42)
random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"

def perform_mutation_km_analysis(mutation_data: MutationData):
    output_dir = "../results/mutation_data_analysis"
    os.makedirs(output_dir, exist_ok=True)

    kaplan_meier_analysis(mutation_data(), output_dir)

def perform_mirna_cox_regression(mirna_data: miRNAData, method: str):
    output_dir = f"../results/mirna_data_analysis/cox_regression_{method}"
    os.makedirs(output_dir, exist_ok=True)

    cph = cox_regression(mirna_data.df_for_survival, output_path = os.path.join(output_dir, 'cox_results.csv'), save_summary=True)
    cox_regression_results(cph, mirna_data.explained_variance, mirna_data(), mirna_data.loadings_df, method=method, output_dir=output_dir, save_significant_mirna=True)

def perform_rna_cox_regression(rna_data: RNAData):
    output_dir = "../results/rna_data_analysis"
    os.makedirs(output_dir, exist_ok=True)

    rna_cox_regression(rna_data(), output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform Kaplan-Meier analysis on mutation data.")
    parser.add_argument("-mu", "--mutation", action="store_true", help="Run the Kaplan-Meier analysis on Mutation Data.")
    parser.add_argument("-mi", "--mirna", action="store_true", help="Run the Cox-Regression analysis on miRNA Data.")
    parser.add_argument("-r", "--rna", action="store_true", help="Run the Cox-Regression analysis on RNASeq Data.")
    parser.add_argument("--method", type=str, help="Specify the method for miRNA Cox-Regression analysis (weighted or simple).")
    args = parser.parse_args()
    
    clinical_data = ClinicalData(Conf.datasets['Clinical'])
    clinical_data.preprocess_data()

    patient_mappings = PatientMappings(clinical_data()['patient_id'])

    clinical_data()['pathologic_stage'].isna().sum()

    if args.mutation:

        print("Starting Kaplan-Meier analysis on Mutation Data...")

        mutation_data = MutationData(Conf.datasets['Mutation'])
        mutation_data.add_stage_data(clinical_data)
        mutation_data.clean_data()

        perform_mutation_km_analysis(mutation_data)

    elif args.mirna:

        method = args.method if args.method in ['simple', 'weighted'] else 'weighted'
        print(f"Starting Cox-Regression analysis on miRNA Data using \"{method}\" method...")

        mirna_data = miRNAData(Conf.datasets['miRNA'])
        mirna_data.add_stage_data(clinical_data)
        mirna_data.clean_data()
        mirna_data.pca(0.85)

        perform_mirna_cox_regression(mirna_data, method)
    
    elif args.rna:

        print("Starting Cox-Regression analysis on RNAseq Data...")

        rna_data = RNAData(Conf.datasets['Reduced_RNASeq'])

        perform_rna_cox_regression(rna_data)
        