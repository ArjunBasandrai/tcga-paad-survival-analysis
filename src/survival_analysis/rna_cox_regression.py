import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
from lifelines.utils import k_fold_cross_validation

from tqdm import tqdm

from sklearn.decomposition import PCA

pd.options.mode.chained_assignment = None

def split_data(df: pd.DataFrame) -> tuple:
    X, y = df.drop(['status', 'overall_survival'], axis=1), df['overall_survival']
    return X, y

def get_top_100_rna_rsf(df: pd.DataFrame, X: pd.DataFrame) -> list:
    y_surv = Surv.from_dataframe("status", "overall_survival", df)

    rsf = RandomSurvivalForest(
        n_estimators=100,
        random_state=42,
        max_features="sqrt",
        min_samples_split=10,
    )
    rsf.fit(X, y_surv)

    baseline_cindex = concordance_index(y_surv["overall_survival"], -rsf.predict(X), y_surv["status"])

    feature_importances = {}

    for col in tqdm(X.columns):
        X_permuted = X.copy()
        X_permuted[col] = np.random.permutation(X_permuted[col])

        permuted_cindex = concordance_index(y_surv["overall_survival"], -rsf.predict(X_permuted), y_surv["status"])

        importance = baseline_cindex - permuted_cindex
        feature_importances[col] = importance

    feature_importances = pd.Series(feature_importances).sort_values(ascending=False)

    top_100_features = feature_importances.nlargest(100).index.to_list()
    return top_100_features

def pca(df: pd.DataFrame, max_explained_variance: float) -> tuple:
    df_rna = df[df.columns[1:-2]]
    X = df_rna.values

    pca = PCA(n_components=max_explained_variance)
    principal_components = pca.fit_transform(X)
    print(f"Number of PCs chosen to capture {int(max_explained_variance * 100)}% variance:", pca.n_components_)

    loadings_df = pd.DataFrame(
        np.abs(pca.components_.T),
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
        index=df_rna.columns
    )

    df_pcs = pd.DataFrame(
        principal_components,
        columns=[f"PC{i+1}" for i in range(principal_components.shape[1])],
        index=df_rna.index
    )
    df_for_survival = pd.concat([df_pcs, df[['overall_survival', 'status']]], axis=1)

    return loadings_df, df_for_survival

def check_epv(df_for_survival: pd.DataFrame, limit: int = 2) -> bool:
    num_events = df_for_survival['status'].sum()
    num_features = len(df_for_survival.columns) - 2

    EPV = num_events / num_features
    print(f"EPV: {EPV:.2f}")

    return EPV >= limit

def find_optimal_regularization_params(df_for_survival: pd.DataFrame) -> tuple:
    param_grid_penalizer = [0.3, 0.35, 0.4]
    param_grid_l1_ratio  = [0.45, 0.5, 0.55]

    best_score = -np.inf
    best_params = (None, None)

    for penalizer in tqdm(param_grid_penalizer, desc="Processing"):
        for l1_ratio in tqdm(param_grid_l1_ratio):
            cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
            
            scores = k_fold_cross_validation(
                cph,
                df_for_survival,
                duration_col='overall_survival',
                event_col='status',
                k=5, 
                scoring_method="concordance_index",
                seed=999
            )
            
            mean_score = np.mean(scores)
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = (penalizer, l1_ratio)

    print(f"Best penalizer: {best_params[0]}")
    print(f"Best l1_ratio:  {best_params[1]}")
    print(f"Best mean CV c-index: {best_score:.4f}")

    return best_params

def fit_cph(df_for_survival: pd.DataFrame, best_params: tuple) -> CoxPHFitter:
    cph = CoxPHFitter(penalizer=best_params[0], l1_ratio=best_params[1])
    cph.fit(df_for_survival, show_progress=False, duration_col='overall_survival', event_col='status')
    return cph

def check_assumptions(cph: CoxPHFitter, df_for_survival: pd.DataFrame, p_value_threshold: float = 0.05) -> None:
    cph.check_assumptions(df_for_survival, p_value_threshold=p_value_threshold, show_plots=False)

def validate_cph(cph: CoxPHFitter, df_for_survival: pd.DataFrame, best_params: tuple) -> bool:
    seeds = np.random.randint(1, 1000001, size=5)
    scores = []

    for seed in tqdm(seeds):
        cph = CoxPHFitter(penalizer=best_params[0], l1_ratio=best_params[1])
        cv_scores = k_fold_cross_validation(
            cph,
            df_for_survival,
            duration_col='overall_survival',
            event_col='status',
            k=5,
            scoring_method="concordance_index",
            seed=seed
        )
        scores.append(np.mean(cv_scores))

    mean_score = np.mean(scores)
    score_variance = np.var(scores)

    print(f"Mean c-index across seeds: {mean_score:.4f}")
    print(f"Variance in c-index across seeds: {score_variance:.4f}")

    return score_variance < 0.01

def extract_important_rna(loadings_df: pd.DataFrame, best_components: list) -> list:
    important_rna = {}

    for component in best_components:
        component_composition = loadings_df[component].sort_values(ascending=False)[:20]
        for rna, value in component_composition.items():
            if rna not in important_rna or value > important_rna[rna]:
                important_rna[rna] = value

    sorted_important_rna = sorted(important_rna.items(), key=lambda x: x[1], reverse=True)
    return [rna for rna, _ in sorted_important_rna]

def save_plots(important_rna: list, df: pd.DataFrame, output_path: str) -> None:
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for gene in important_rna:
        correlation = df[[gene, 'overall_survival']].corr().iloc[0, 1]
        plt.figure(figsize=(8, 6))
        plt.scatter(df[gene], df['overall_survival'], alpha=0.7, color="salmon")
        plt.title(f"{gene} vs. Overall Survival (corr={correlation:.2f})")
        plt.xlabel(gene)
        plt.ylabel("Overall Survival")
        plt.grid(True)
        plt.savefig(os.path.join(output_path, f"{gene}_vs_Overall_Survival.png"), dpi=300, bbox_inches="tight")
        plt.close()

def prepare_data(df: pd.DataFrame, important_rna: list) -> tuple:
    threshold = 365

    X = df[important_rna]
    df['binned_overall_survival'] = pd.cut(x=df['overall_survival'], 
            bins=[-float('inf'), threshold, float('inf')], 
            labels=[0, 1]
            )

    X_train, X_test, y_train, y_test = train_test_split(
        X, df['binned_overall_survival'], test_size=0.125, random_state=42
    )

    return X_train, X_test, y_train, y_test

def rna_cox_regression(df: pd.DataFrame, output_path: str) -> None:
    X, _ = split_data(df)
    top_100_features = get_top_100_rna_rsf(df, X)
    # top_100_features = ['FAM25A', 'SFTA2', 'KRT13', 'SFTPA2', 'TMEM84', 'ZNF266', 'SPP1', 'ZFP57', 'OTUD7A', 'CBFA2T2', 'LOC256880', 'KIF20A', 'LHB', 'RAB40AL', 'SIAH3', 'SLC26A10', 'OR2A1', 'WNT7B', 'FAM114A1', 'CPZ', 'PPP2R3A', 'NEIL1', 'PRC1', 'TAL1', 'HIST1H2BK', 'KCTD3', 'CXCL9', 'CD99L2', 'ZNF835', 'C2orf70', 'MT2A', 'SH2D4A', 'TKT', 'POU5F1', 'PML', 'HIST1H1C', 'CCDC6', 'FHOD3', 'PVRL3', 'IP6K2', 'CD300LG', 'IL8', 'SH3BP2', 'BST2', 'RAB39B', 'CLEC4G', 'DHRS9', 'PCSK2', 'SLK', 'LY6D', 'C6orf114', 'CGB', 'UNC13D', 'DMPK', 'MMP10', 'FADS3', 'CLEC5A', 'LOC642597', 'NOL4', 'PDE1C', 'ALPPL2', 'TNS4', 'ZBTB38', 'SEMA3C', 'PPP1R15A', 'LOC100272228', 'MGAT1', 'GCK', 'SPINK7', 'RFPL2', 'STK33', 'LMNB1', 'ZNF233', 'KLHDC7B', 'TMC2', 'FBN3', 'WDR88', 'NPR3', 'PCDHA9', 'ERAP2', 'ANXA1', 'UPF1', 'IL32', 'CDK3', 'PGM1', 'GPR77', 'HBEGF', 'GPR111', 'NMD3', 'LONRF2', 'STAT1', 'LY75', 'ANLN', 'PAX7', 'C1orf213', 'DCUN1D2', 'ALDH3A1', 'BMPER', 'PRR5-ARHGAP8', 'LARGE']

    df = df[top_100_features + ['status', 'overall_survival']]
    loadings_df, df_for_survival = pca(df, 0.85)

    if not check_epv(df_for_survival, 2):
        print("Warning: The EPV is less than 2. The model may not be stable.")
        print("Exiting...")
        return

    best_params = find_optimal_regularization_params(df_for_survival)

    cph = fit_cph(df_for_survival, best_params)
    check_assumptions(cph, df_for_survival)

    cph_fine = validate_cph(cph, df_for_survival, best_params)
    if not cph_fine:
        print("Warning: The model may not be stable. Please check the variance in c-index scores.")
        print("Exiting...")
        return
    
    best_components = cph.summary.query("p < 0.05").index.to_list()
    important_rna = extract_important_rna(loadings_df, best_components)

    save_plots(important_rna, df, output_path)

    X_train, X_test, y_train, y_test = prepare_data(df, important_rna)

    rf = RandomForestClassifier(random_state=999)
    rf.fit(X_train, y_train)

    print(f"Train accuracy: {rf.score(X_train, y_train)*100:.2f}%")
    print(f"Test accuracy:  {rf.score(X_test, y_test)*100:.2f}%")

    report = classification_report(y_test, rf.predict(X_test), target_names=['Low Survival', 'High Survival'])
    print(report)
