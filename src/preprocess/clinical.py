import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer

class ClinicalData:
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
    
    def select_cols(self) -> None:
        self.df.drop(['ethnicity', 'race', 'overallsurvival', 'status'], axis=1, inplace=True)
        
    def fill_na_clinical_data(self) -> None:
        def fill_na(row: pd.Series) -> pd.Series:
            if pd.isnull(row['pathology_N_stage']):
                if row['pathology_T_stage'] == 't0':
                    row['pathology_N_stage'] = 'n0'
                elif row['pathology_T_stage'] in ['t1', 't2']:
                    row['pathology_N_stage'] = 'n0'
                else:
                    row['pathology_N_stage'] = 'n1'
                    
            if pd.isnull(row['pathology_M_stage']):
                if row['pathology_T_stage'] in ['t0', 't1']:
                    row['pathology_M_stage'] = 'm0'
                ## needs improvement ->
                else:
                    row['pathology_M_stage'] = 'm1'
    
            if pd.isnull(row['number_of_lymph_nodes']) and row['pathology_N_stage'] == 'n0':
                row['number_of_lymph_nodes'] = 0
                
            return row
            
        self.df['pathologic_stage'] = self.df['pathologic_stage'].fillna('stage0')
        self.df['pathology_T_stage'] = self.df['pathology_T_stage'].fillna('t0')
        self.df['radiation_therapy'] = self.df['radiation_therapy'].fillna('no')
        self.df['residual_tumor'] = self.df['residual_tumor'].fillna('r0')
    
        self.df = self.df.apply(fill_na, axis=1)
            
    def label_encode_clinical_data(self) -> None:
        categorical_columns = ['pathologic_stage', 'pathology_T_stage', 'pathology_N_stage',
                              'pathology_M_stage', 'histological_type', 'gender', 'radiation_therapy',
                              'residual_tumor']
    
        for column in categorical_columns:
            if column in self.df.columns:
                unique_values = [value for value in self.df[column].unique() if pd.notna(value)]
                self.label_mappings[column] = {value: idx for idx, value in enumerate(sorted(unique_values))}
    
        def to_id(row: pd.Series) -> pd.Series:
            for column in categorical_columns:
                if column in row.index and pd.notna(row[column]):
                    row[column] = self.label_mappings[column][row[column]]
            return row
    
        self.df = self.df.apply(to_id, axis=1)
    
    def impute_na_clinical_data(self) -> None:
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.df['histological_type'] = categorical_imputer.fit_transform(self.df[['histological_type']])
        self.df['histological_type'] = self.df['histological_type'].astype(int)
    
        knn_imputer = KNNImputer(n_neighbors=5)
        numerical_columns = ['overall_survival', 'number_of_lymph_nodes']
        self.df[numerical_columns] = self.df[numerical_columns].astype(float)    
        self.df[numerical_columns] = knn_imputer.fit_transform(self.df[numerical_columns])
        self.df[numerical_columns] = self.df[numerical_columns].astype(int)    

    def preprocess_data(self) -> None:
        self.select_cols()
        self.fill_na_clinical_data()
        self.label_encode_clinical_data()
        self.impute_na_clinical_data()

    def get_label_decoded(self) -> pd.DataFrame:
        df_copy = self.df.copy(deep=True)
        for column, mapping in self.label_mappings.items():
            inv_mapping = {v: k for k, v in mapping.items()}
            df_copy[column] = df_copy[column].apply(
                lambda x: inv_mapping[x] if (pd.notnull(x) and x in inv_mapping) else x
            )
        return df_copy

    def __call__(self) -> pd.DataFrame:
        return self.df