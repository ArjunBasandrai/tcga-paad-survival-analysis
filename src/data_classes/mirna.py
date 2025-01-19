import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from data_classes.clinical import ClinicalData

class miRNAData:
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
        print("Starting data cleaning. Shape of un-clean data:", self.df.shape)

        for col in self.df.columns[1:]:
            self.df[col] = self.df[col].astype(int)

        def drop_identical_columns() -> None:
            def find_identical_columns(dataframe: pd.DataFrame) -> dict:
                identical_columns = {}
                for col1 in dataframe.columns:
                    for col2 in dataframe.columns:
                        if col1 != col2 and dataframe[col1].equals(dataframe[col2]):
                            identical_columns.setdefault(col1, []).append(col2)
                return identical_columns
            
            identical_columns = find_identical_columns(self.df)
            
            columns_to_drop = set()
            for duplicates in identical_columns.values():
                columns_to_drop.update(duplicates)
            
            return self.df.drop(columns=list(columns_to_drop), axis=1)

        def drop_low_var_columns(threshold: float = 0.01) -> pd.DataFrame:
            cols = self.df.columns[1:-2]
            columns_to_drop = list()
            for col in cols:
                events = self.df['status'].astype(bool)
                p_var  = self.df.loc[events, col].var()
                n_var = self.df.loc[~events, col].var()
                if p_var <= threshold or n_var <= threshold:
                    columns_to_drop.append(col)
            return self.df.drop(columns=columns_to_drop, axis=1)
        
        self.df = drop_identical_columns()
        self.df = drop_low_var_columns()

        print("Data cleaning complete. Shape of cleaned data:", self.df.shape)

    def add_stage_data(self, clinical_data: ClinicalData) -> None:
        self.merge_data(clinical_data)
        self.select_common_patients()
    
    def pca(self, max_explained_variance: float) -> None:
        self.mirna_cols = self.df.columns[1:-2]
        df_mirna = self.df[self.mirna_cols]

        pca = PCA(n_components=max_explained_variance)
        self.principal_components = pca.fit_transform(df_mirna.values)
        self.explained_variance = pca.explained_variance_ratio_

        loadings = pca.components_
        self.loadings_df = pd.DataFrame(
            loadings.T,
            index=df_mirna.columns,
            columns=[f"PC{i+1}" for i in range(loadings.shape[0])]
        )

        print(f"Number of PCs chosen to capture {max_explained_variance * 100:.2f}% variance:", pca.n_components_)

        df_pcs = pd.DataFrame(
            self.principal_components,
            columns=[f"PC{i+1}" for i in range(self.principal_components.shape[1])],
            index=df_mirna.index
        )
        self.df_for_survival = pd.concat([df_pcs, self.df[['overall_survival', 'status']]], axis=1)

    def __call__(self) -> pd.DataFrame:
        return self.df