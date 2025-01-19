import pandas as pd

import os
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_classes.clinical import ClinicalData

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
        
    def __call__(self) -> pd.DataFrame:
        return self.df