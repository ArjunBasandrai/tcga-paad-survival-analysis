import pandas as pd

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

    def add_stages(self, clinical_data: ClinicalData) -> None:
        self.df = self.df.merge(
            clinical_data()[['patient_id', 'pathologic_stage']],
            on='patient_id',
            how='left'
        )

    def merge_stages(self) -> None:
        self.df['pathologic_stage'] = self.df['pathologic_stage'].replace({3:2, 4:2})
        self.label_mappings = {
            'no cancer': 0,
            'early stage cancer': 1,
            'late stage cancer': 2,
        }

    def clean_data(self) -> None:
        for col in self.df.columns[1:]:
            self.df[col] = self.df[col].astype(int)

        self.df = self.df[self.df['pathologic_stage'] != 0]

    def add_stage_data(self, clinical_data: ClinicalData) -> None:
        self.add_stages(clinical_data)
        self.merge_stages()
        
    def __call__(self) -> pd.DataFrame:
        return self.df