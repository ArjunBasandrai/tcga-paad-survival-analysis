import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class RNAData:
    def __init__(self, df_path: str) -> None:
        self.df = pd.read_csv(df_path)
        self.df.drop(['Unnamed: 0'], axis=1, inplace=True)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.rna_cols = self.df.columns[1:-2]

    def normalize_data(self) -> None:
        self.df[self.rna_cols] = self.scaler.fit_transform(self.df[self.rna_cols])

    def bin_survival(self, n_bins: int) -> None:
        self.df['binned_overall_survival'] = pd.cut(x=self.df['overall_survival'], bins=n_bins, labels=list(range(1, n_bins + 1)))
        self.rna_cols = self.df.columns[1:-3]

    def __call__(self) -> pd.DataFrame:
        return self.df