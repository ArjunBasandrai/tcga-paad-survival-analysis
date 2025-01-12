import argparse
import os

from preprocess.config.config import Conf
from preprocess.clinical import ClinicalData
from preprocess.mutation import MutationData
from preprocess.mirna import miRNAData
from preprocess.config.patient_mappings import PatientMappings

def perform_km_analysis(mutation_data: MutationData):
    output_dir = "../results/kaplan_meier_analysis"
    os.makedirs(output_dir, exist_ok=True)

    mutation_data.kaplan_meier_analysis(output_dir)

def perform_cox_regression(mirna_data: miRNAData, method: str):
    output_dir = f"../results/cox_regression/cox_regression_{method}"
    os.makedirs(output_dir, exist_ok=True)

    mirna_data.cox_regression(output_path = os.path.join(output_dir, 'cox_results.csv'), save_summary = True, method=method)
    mirna_data.cox_regression_results(output_dir = output_dir, save_significant_mirna = True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform Kaplan-Meier analysis on mutation data.")
    parser.add_argument("-k", "--km", action="store_true", help="Run the Kaplan-Meier analysis.")
    parser.add_argument("-c", "--cx", action="store_true", help="Run the Cox-Regression analysis.")
    parser.add_argument("--method", type=str, help="Specify the method for Cox-Regression analysis (weighted or simple).")
    args = parser.parse_args()

    clinical_data = ClinicalData(Conf.datasets['Clinical'])
    clinical_data.preprocess_data()

    patient_mappings = PatientMappings(clinical_data()['patient_id'])

    clinical_data()['pathologic_stage'].isna().sum()

    if args.km:

        print("Starting Kaplan-Meier analysis...")

        mutation_data = MutationData(Conf.datasets['Mutation'])
        mutation_data.add_stage_data(clinical_data)
        mutation_data.clean_data()

        perform_km_analysis(mutation_data)

    elif args.cx:

        method = args.method if args.method in ['simple', 'weighted'] else 'weighted'
        print(f"Starting Cox-Regression analysis using \"{method}\" method...")

        mirna_data = miRNAData(Conf.datasets['miRNA'])
        mirna_data.add_stage_data(clinical_data)
        mirna_data.clean_data()
        mirna_data.pca(0.85)

        perform_cox_regression(mirna_data, method)