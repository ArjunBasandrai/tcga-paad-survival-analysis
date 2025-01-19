import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

from scipy.stats import gaussian_kde

def _cox_regression_preprocess(df_for_survival: pd.DataFrame) -> pd.DataFrame:
    # Potential non-linear component identified using cph.check_assumptions(...)
    df_for_survival['PC25**2'] = (df_for_survival['PC25'] - df_for_survival['PC25'].mean()) ** 2
    return df_for_survival

def _fit_cox_regression(df_for_survival: pd.DataFrame):
    cph = CoxPHFitter()
    cph.fit(df_for_survival, show_progress=True, duration_col="overall_survival", event_col="status")
    return cph

def _check_cph_assumptions(cph, df_for_survival: pd.DataFrame, p_value: float, show_plots: bool = False) -> None:
    cph.check_assumptions(df_for_survival, p_value_threshold=p_value, show_plots=show_plots)

def _save_cph_regression(cph, output_path: str) -> None:
    cph.summary.to_csv(output_path)

def cox_regression(df_for_survival: pd.DataFrame, output_path: str, save_summary: bool = False, check_assumptions: bool = True, p_value: float = 0.05) -> None:
    df_for_survival = _cox_regression_preprocess(df_for_survival)
    cph = _fit_cox_regression(df_for_survival)

    if check_assumptions:
        _check_cph_assumptions(cph, df_for_survival, p_value=p_value)

    if save_summary and output_path:
        _save_cph_regression(cph, output_path)
    
    return cph

def _extract_important_mirna(cph, explained_variance, loadings_df: pd.DataFrame, method: str, p_value_threshold: float = 0.05) -> np.array:
    important_components = [int(component[2:]) for component in list(cph.summary.query(f"p < {p_value_threshold}").index)]
    weights = {}
    for component in important_components:
        component_explained_variance = explained_variance[component - 1]
        significant_features = loadings_df[f"PC{component}"].abs().nlargest(8).reset_index()
        for i in range(len(significant_features)):
            feature = significant_features.iloc[i]
            feature_name = feature['index']
            feature_value = feature[f"PC{component}"]
            if feature_name not in weights.keys():
                weights[feature_name] = component_explained_variance * feature_value if method == 'weighted' else feature_value
            else:
                weights[feature_name] = (weights[feature_name] + component_explained_variance * feature_value) if method == 'weighted' else max(weights[feature_name], feature_value)
    sorted_mirna = np.array(sorted(weights.items(), key=lambda item: item[1], reverse=True))
    if len(sorted_mirna) > 15:
        sorted_mirna = sorted_mirna[:15, 0]
    return sorted_mirna

def _plot_important_mirna_distribution(df: pd.DataFrame, important_mirna: np.array, output_dir: str) -> None:
    fig, axes = plt.subplots(5, 3, figsize=(15, 20))
    axes = axes.flatten()

    for idx, mirna in enumerate(important_mirna):
        mirna_values = df[mirna]
        mirna_counts = mirna_values.value_counts().reset_index()
        mirna_counts.columns = [mirna, 'count']
        mirna_counts = mirna_counts.sort_values(by=mirna)

        x_values = mirna_counts[mirna]
        y_values = mirna_counts['count']

        kde = gaussian_kde(mirna_values)
        kde_x = np.linspace(x_values.min(), x_values.max(), 100)
        kde_y = kde(kde_x)
        scaled_kde_y = kde_y * (y_values.max() / kde_y.max())

        axes[idx].plot(kde_x, scaled_kde_y, color='teal', label='KDE')
        axes[idx].fill_between(kde_x, scaled_kde_y, color='teal', alpha=0.3)
        axes[idx].plot(x_values, y_values, color='deeppink', label='Line Plot')

        axes[idx].set_title(f"{mirna}")
        axes[idx].grid(alpha=0.3)
        axes[idx].set_xlabel("Value")
        axes[idx].set_ylabel("Count / Density")
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].legend()

    output_dir = os.path.join(output_dir, "important_miRNA_distribution")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "mirna_distribution.png")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close(fig)

def _plot_kaplan_meier_curves(df: pd.DataFrame, important_mirna: np.array, output_dir: str, p_value_threshold: float = 0.05) -> pd.DataFrame:
    columns = list(important_mirna)
    columns_to_transform = columns[:]
    columns.extend(['status', 'overall_survival'])
    df_mirna = df[columns]

    for col in columns_to_transform:
        col_quantiles = df_mirna[col].quantile([0.25, 0.75]).values
        if col_quantiles[0] == col_quantiles[1]:
            continue
        df_mirna.loc[:, col] = pd.cut(
            df_mirna[col], 
            bins=[-float('inf'), col_quantiles[0], col_quantiles[1], float('inf')], 
            labels=[0, 1, 2]
        )
    
    significant_mirna = []

    output_dir = os.path.join(output_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    for mirna in tqdm(df_mirna.columns[0:-2]):
        mirna_low = df_mirna[df_mirna[mirna] == 0]
        mirna_medium = df_mirna[df_mirna[mirna] == 1]
        mirna_high = df_mirna[df_mirna[mirna] == 2]

        min_group_size = 10
        if len(mirna_low) < min_group_size or len(mirna_medium) < min_group_size or len(mirna_high) < min_group_size:
            continue

        results_low_medium = logrank_test(
            mirna_low["overall_survival"], mirna_medium["overall_survival"],
            event_observed_A=mirna_low["status"], event_observed_B=mirna_medium["status"]
        )
        results_low_high = logrank_test(
            mirna_low["overall_survival"], mirna_high["overall_survival"],
            event_observed_A=mirna_low["status"], event_observed_B=mirna_high["status"]
        )
        results_medium_high = logrank_test(
            mirna_medium["overall_survival"], mirna_high["overall_survival"],
            event_observed_A=mirna_medium["status"], event_observed_B=mirna_high["status"]
        )

        if (results_low_medium.p_value < p_value_threshold or
                results_low_high.p_value < p_value_threshold or
                results_medium_high.p_value < p_value_threshold):
            significant_mirna.append({
                "mirna": mirna,
                "low_medium_p_value": results_low_medium.p_value,
                "low_high_p_value": results_low_high.p_value,
                "medium_high_p_value": results_medium_high.p_value
            })

            kmf = KaplanMeierFitter()
            plt.figure(figsize=(10, 6))
            
            kmf.fit(mirna_low["overall_survival"], event_observed=mirna_low["status"], label="Low")
            kmf.plot_survival_function()
            
            kmf.fit(mirna_medium["overall_survival"], event_observed=mirna_medium["status"], label="Medium")
            kmf.plot_survival_function()
            
            kmf.fit(mirna_high["overall_survival"], event_observed=mirna_high["status"], label="High")
            kmf.plot_survival_function()
            
            plt.title(f"Kaplan-Meier Survival Curve for {mirna}")
            plt.xlabel("Time (days)")
            plt.ylabel("Survival Probability")
            plt.legend()
            plt.grid(True)

            plot_filename = os.path.join(output_dir, f"{mirna}_km_curve_paad.png")
            plt.savefig(plot_filename, dpi=300)
            plt.close()

    return pd.DataFrame(significant_mirna)
            
def cox_regression_results(cph, explained_variance, df: pd.DataFrame, loadings_df: pd.DataFrame, method: str, output_dir: str, save_significant_mirna: bool = False) -> None:
    important_mirna = _extract_important_mirna(cph, explained_variance, loadings_df, method)
    _plot_important_mirna_distribution(df, important_mirna, output_dir)
    significant_mirna = _plot_kaplan_meier_curves(df, important_mirna, output_dir)
    if save_significant_mirna:
        significant_mirna.to_csv(os.path.join(output_dir, "significant_mirna.csv"))