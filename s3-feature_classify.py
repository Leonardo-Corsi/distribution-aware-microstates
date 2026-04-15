
#%% Import 
# basic
import argparse
import json
import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from datetime import datetime, timezone
import hashlib
# plotting
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.switch_backend('Agg')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    #'mathtext.fontset' : 'stix',
    'font.size': 10
})
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
colors = {'normal train':'#0072B2',
          'normal eval':'#56B4E9',
          'abnormal train':'#D55E00',
          'abnormal eval':'#F0E442'}
# classification, stats, performance
from dcurves import dca,plot_graphs
from scipy.stats import mannwhitneyu
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from _config_loader import load_configs
plt.close('all')
plt.ioff()

#%% Loading and adaptation of dataframes
def _parse_args():
    parser = argparse.ArgumentParser(description="S3 feature classification (config-driven runtime).")
    parser.add_argument("--config", required=True, help="Path to config file.")
    parser.add_argument(
        "--recompute-studies",
        action="store_true",
        help="Force Optuna study recomputation instead of loading cached studies.",
    )
    return parser.parse_args()


def _resolve_path(base_dir, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty path string, got: {value!r}")
    expanded = os.path.expanduser(value)
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    return os.path.normpath(os.path.join(base_dir, expanded))


ARGS = _parse_args()
CFG = load_configs(config_path=ARGS.config)
S3_CFG = CFG.get("s3", {})
S3_INPUT_STAGE = str(S3_CFG.get("input_stage", "with_ica")).strip().lower()
if S3_INPUT_STAGE not in {"with_ica", "without_ica"}:
    raise ValueError(
        f"Unsupported cfg['s3']['input_stage']: {S3_INPUT_STAGE!r}. Use 'with_ica' or 'without_ica'."
    )

if "output_subdir" in S3_CFG and str(S3_CFG.get("output_subdir", "")).strip():
    output_subdir_value = str(S3_CFG.get("output_subdir"))
elif S3_INPUT_STAGE == "with_ica":
    output_subdir_value = str(S3_CFG.get("output_subdir_with_ica", "with_ica"))
else:
    output_subdir_value = str(S3_CFG.get("output_subdir_without_ica", "without_ica"))

OUTPUT_BASE_DIR = _resolve_path(CFG["s3"]["s3_parent"], output_subdir_value)
STUDIES_DIR = _resolve_path(
    OUTPUT_BASE_DIR,
    str(S3_CFG.get("studies_subdir", "studies")),
)
INTERMEDIATE_FEATURES_DIR = _resolve_path(
    OUTPUT_BASE_DIR,
    str(S3_CFG.get("intermediate_features_subdir", "intermediate-features")),
)
PAPERITEMS_DIR = _resolve_path(
    OUTPUT_BASE_DIR,
    str(S3_CFG.get("paperitems_subdir", "paperitems")),
)
TABLES_DIR = _resolve_path(
    OUTPUT_BASE_DIR,
    str(S3_CFG.get("tables_subdir", "tables")),
)

os.makedirs(STUDIES_DIR, exist_ok=True)
os.makedirs(INTERMEDIATE_FEATURES_DIR, exist_ok=True)
os.makedirs(PAPERITEMS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

STUDY_MANIFEST_PATH = os.path.join(
    STUDIES_DIR,
    str(S3_CFG.get("study_manifest_filename", "study_manifest.json")),
)
STUDY_RUN_AUDIT = []


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def _json_dump_atomic(path, payload):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _config_fingerprint(cfg):
    normalized = json.dumps(cfg, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _study_meta_path(study_name):
    return os.path.join(STUDIES_DIR, f"study_{study_name}.meta.json")


def _extract_study_best(study):
    if study is None:
        return {"best_value": None, "best_params": None}
    try:
        return {
            "best_value": float(study.best_value),
            "best_params": dict(study.best_trial.params),
        }
    except Exception:
        return {"best_value": None, "best_params": None}


def _write_study_metadata(study_name, study_path, study_obj, lifecycle, started_at, ended_at):
    best = _extract_study_best(study_obj)
    meta = {
        "study_name": study_name,
        "study_path": os.path.normpath(study_path),
        "metadata_version": 1,
        "lifecycle": lifecycle,
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "recompute_studies_cli": bool(RECOMPUTE_STUDIES),
        "n_trials": int(len(study_obj.trials)) if study_obj is not None else 0,
        "best_value": best["best_value"],
        "best_params": best["best_params"],
        "config_path": CFG["meta"]["config_path"],
        "config_fingerprint_sha256": _config_fingerprint(CFG),
    }
    _json_dump_atomic(_study_meta_path(study_name), meta)
    STUDY_RUN_AUDIT.append(meta)
    _json_dump_atomic(
        STUDY_MANIFEST_PATH,
        {
            "run_started_at_utc": RUN_STARTED_AT_UTC,
            "run_last_updated_utc": _utc_now_iso(),
            "entries": STUDY_RUN_AUDIT,
        },
    )

PATH_AVGPSD = os.path.join(INTERMEDIATE_FEATURES_DIR, "df_avgPSD.parquet")
PATH_EDF_MTMI = os.path.join(INTERMEDIATE_FEATURES_DIR, "df_EDF_MTMI.parquet")
PATH_EDF_MSD = os.path.join(INTERMEDIATE_FEATURES_DIR, "df_EDF_MSD.parquet")
PATH_AVGMSD = os.path.join(INTERMEDIATE_FEATURES_DIR, "df_avgMSD.parquet")

INTERMEDIATE_DF_PATHS = {
    "avg_psd": PATH_AVGPSD,
    "edf_mtmi": PATH_EDF_MTMI,
    "edf_msd": PATH_EDF_MSD,
    "avg_msd": PATH_AVGMSD,
}


def _resolve_s2_feature_dir():
    extraction_cfg = CFG.get("s2", {}).get("extraction", {})
    if S3_INPUT_STAGE == "with_ica":
        subdir = str(extraction_cfg.get("output_subdir_with_ica", "features_with_ica"))
    elif S3_INPUT_STAGE == "without_ica":
        subdir = str(extraction_cfg.get("output_subdir_without_ica", "features_without_ica"))
    return _resolve_path(CFG["s2"]["s2_parent"], subdir)


def _load_raw_feature_tables():
    s2_dir = _resolve_s2_feature_dir()
    path_mtmi = os.path.join(s2_dir, "df_MTMI.parquet")
    path_psd = os.path.join(s2_dir, "df_PSD.parquet")
    path_msd = os.path.join(s2_dir, "df_MSD.parquet")
    missing = [p for p in [path_mtmi, path_psd, path_msd] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing required S2 parquet inputs for S3 run. Missing paths:\n"
            + "\n".join(missing)
        )
    print(f"[S3] Loading S2 parquet inputs from: {s2_dir}")
    return pd.read_parquet(path_mtmi), pd.read_parquet(path_psd), pd.read_parquet(path_msd)

RECOMPUTE_STUDIES = bool(ARGS.recompute_studies) or bool(S3_CFG.get("force_recompute_studies", False))
RUN_STARTED_AT_UTC = _utc_now_iso()
recompute_dfs = bool(S3_CFG.get("recompute_intermediate_dfs", False)) or any(
    not os.path.exists(path) for path in INTERMEDIATE_DF_PATHS.values()
)
print(f"[S3] output_base={OUTPUT_BASE_DIR}")
print(f"[S3] studies_dir={STUDIES_DIR}")
print(f"[S3] intermediate_dir={INTERMEDIATE_FEATURES_DIR}")
print(f"[S3] paperitems_dir={PAPERITEMS_DIR}")
print(f"[S3] tables_dir={TABLES_DIR}")
print(f"[S3] recompute_intermediate_dfs={recompute_dfs}")
print(f"[S3] recompute_studies={RECOMPUTE_STUDIES}")

if recompute_dfs:
    df_MTMI, df_PSD, df_MSD = _load_raw_feature_tables()

    print("Loading PSD data...")
    df_avgPSD = df_PSD.groupby(["Group", "ID", "f"])["PSD"].mean().reset_index()
    df_avgPSD.rename(columns={"PSD": "average_PSD"}, inplace=True)
    df_avgPSD["average_PSD"] = df_avgPSD["average_PSD"] * 1e12
    # Keep PSD in linear uV^2/Hz plus explicit dB transform for plotting/classification.
    df_avgPSD["average_PSD_dB"] = 10.0 * np.log10(
        np.maximum(df_avgPSD["average_PSD"].to_numpy(dtype=float), np.finfo(float).tiny)
    )
    df_avgPSD.to_parquet(PATH_AVGPSD, index=False)

    print("Loading MTMI data and computing EDF...")
    max_time_ms = 300
    df_MTMI = df_MTMI.query("`MTMI Time [ms]` < @max_time_ms")
    tau = 1000 / 128
    bins = np.arange(0, max_time_ms + tau, tau) - tau / 2
    k_values = np.arange(len(bins) - 1)
    edf_mtmi_list = []
    for (group, id_), group_df in tqdm(df_MTMI.groupby(["Group", "ID"]), desc="MTMI EDF"):
        hist, _ = np.histogram(group_df["MTMI Time [ms]"], bins=bins, density=True)
        edf_mtmi_list.append(pd.DataFrame({"Group": group, "ID": id_, "k": k_values, "density": hist}))
    df_EDF_MTMI = pd.concat(edf_mtmi_list, ignore_index=True)
    df_EDF_MTMI = df_EDF_MTMI[df_EDF_MTMI["k"] >= 4]
    df_EDF_MTMI.to_parquet(PATH_EDF_MTMI, index=False)

    print("Loading MSD data and computing EDF...")
    max_time_ms = 120
    max_duration = np.ceil(max_time_ms / tau)
    df_MSD = df_MSD.query("Duration < @max_duration")
    df_MSD = df_MSD.query("map != -1")
    bins = np.arange(0, max_duration, 1) - 0.5
    edf_ms_list = []
    for (group, id_, map_), group_df in tqdm(df_MSD.groupby(["Group", "ID", "map"]), desc="MSD EDF"):
        hist, _ = np.histogram(group_df["Duration"], bins=bins, density=True)
        k_values = np.arange(len(hist))
        edf_ms_list.append(pd.DataFrame({"Group": group, "ID": id_, "map": map_, "k": k_values, "density": hist}))
    df_EDF_MSD = pd.concat(edf_ms_list, ignore_index=True)
    df_EDF_MSD = df_EDF_MSD.query("k >= 3")
    map_mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
    df_EDF_MSD["map"] = df_EDF_MSD["map"].map(map_mapping)
    df_EDF_MSD.to_parquet(PATH_EDF_MSD, index=False)

    print("Computing MSD-avg data...")
    df_avgMSD = df_MSD.groupby(["Group", "ID", "map"]).mean().reset_index()
    df_avgMSD["map"] = df_avgMSD["map"].map(map_mapping)
    df_avgMSD.to_parquet(PATH_AVGMSD, index=False)



#%% Evaluation pipeline func defs
# function defs

def shap_plot(shap_callable,*posargs,ax=None,**kwargs):
    if ax is None: 
        fig,ax = plt.subplots(111)
    plt.sca(ax)
    if 'show' in kwargs.keys(): kwargs.pop('show') # always False
    shap_callable(*posargs,show=False,**kwargs)
    if ax is None:
        fig,ax = plt.gcf(),plt.gca()
        
    if 'title' in kwargs.keys():
        ax.set_title(kwargs['title'])
    plt.tight_layout()
    # plt.show((block=False)
    
    return plt.gca()

def statistics_featuregroup(dataset_long,
                          x_col,y_col,
                          name=None,
                          x_name=None,
                          y_name=None,
                          viz_dB=False,
                          compress_errorbars_ax=False):
    if name is None: name = f"{y_col}-vs-{x_col}"
    if x_name is None: x_name = x_col
    if y_name is None: y_name = y_col
    
    
    

    # ------ Visualize feature group ------
    if compress_errorbars_ax:
        fig_mw, axs = plt.subplot_mosaic([['feat'],['p']],
                                        figsize=(10, 3),
                                        num=name,sharex=True)
    else:
        fig_mw, axs = plt.subplot_mosaic([['feat'],['feat'],['p']],
                                        figsize=(10, 6),
                                        num=name,sharex=True)
    fig_mw.set_label(str(name))
    ax_feat, ax_p = axs['feat'],axs['p'],#axs['u']
    
    dataset_long_train = dataset_long.query("Group == 'abnormal train' or Group == 'normal train'").copy()
    if viz_dB:
        dataset_long_train[y_col] = 10 * np.log10(dataset_long_train[y_col])
    
    sns.lineplot(data=dataset_long_train,
                x=x_col, y=y_col,hue='Group', 
                # errorbars
                estimator=np.median, errorbar=('pi', 50),# 50% percentile interval (IQR)
                err_style = 'bars', 
                err_kws={
                    'capsize': 6,
                    'capthick': 3,
                    'elinewidth': 1.5,
                },
                # lines connecting estimators
                linestyle='', marker='d', markersize=6,
                # other args
                ax=ax_feat, legend=True,palette=colors,
    )
    
    # Update legend entries
    new_legend_labels = {'normal train': 'Normal (train)',
                         'abnormal train': 'Abnormal (train)'
                        }
    for text in ax_feat.get_legend().texts:
        text.set_text(new_legend_labels[text.get_text()])
    # Errobar adjustments
    for child in ax_feat.get_children():
        if isinstance(child, (mpl.collections.LineCollection,
                            mpl.spines.Spine,
                            mpl.lines.Line2D)):
            child.set_alpha(0.6)
    # Decorators
    ax_feat.set_ylabel(y_name)
    ax_feat.grid(axis="x", linestyle="--", alpha=0.5, linewidth=.5, which='both')
    # plt.show((block=False)
    
    
    # ------ Dataset in pivot format ------
    dataset = dataset_long.pivot_table(index=['Group', 'ID'],
                            columns=x_col, values=y_col
                            ).reset_index()
    # If the are missing values in the dataset, impute from group median
    # without imputing the ID and Group columns
    for col in dataset.columns:
        if col not in ['Group', 'ID']:
            dataset[col] = dataset[col].fillna(dataset.groupby('Group')[col].transform('median'))
            
    # ------ Mann-Whitney U test, column-wise ------
    u_values = []
    p_values = []
    for col in dataset.columns[2:]:  # Skip 'Group' and 'ID' columns
        normal_train = dataset.query("Group == 'normal train'")[col]
        abnormal_train = dataset.query("Group == 'abnormal train'")[col]
        u, p = mannwhitneyu(normal_train, abnormal_train, alternative='two-sided')
        u_values.append(u)
        p_values.append(p)
    # Plot p-values
    p_values = np.array(p_values) 
    is_sig = p_values < 0.001/len(p_values) # Bonferroni
    ax_p.plot(dataset.columns[2:][is_sig], p_values[is_sig],
              marker='d', linestyle='', color='k')
    ax_p.plot(dataset.columns[2:][~is_sig], p_values[~is_sig],
              marker='d', linestyle='', color='k', fillstyle='none')
    ax_p.axhline(y=0.001/len(p_values), # Bonferroni
                 color='r', linestyle='--', label='p=0.001')
    ax_p.set_ylabel('Mann-Whitney p')
    ax_p.set_xlabel(x_name)
    ax_p.set_yscale('log')
    ax_p.grid(axis="x", linestyle="--", alpha=0.5, linewidth=.5, which='both')
    # plt.show((block=False)
    
    # Filter columns based on significance
    columns_sig = dataset.columns[:2].tolist() + dataset.columns[2:][is_sig].tolist()
    dataset_sig = dataset[columns_sig].copy()
    
    return dataset_sig

def classifier_SVM_optimized(X_train_scaled, y_train, X_eval_scaled, y_eval, study_path=None, recompute_study=False):
    seed = 0
    np.random.seed(seed)
    study = None
    lifecycle = "optimized"
    
    if study_path is not None and not recompute_study:
        try:
            with open(study_path, 'rb') as f:
                study = pickle.load(f)
            #print("Study loaded successfully.")
            lifecycle = "loaded"
        except Exception as e:
            print(f"Error loading study: {e}, going to optimization")
            study = None
            lifecycle = "load_failed_then_optimized"
        
    if not study:
        print(f"Optimizing SVM parameters: {study_path}...")
        if recompute_study:
            lifecycle = "recomputed"
        elif lifecycle != "load_failed_then_optimized":
            lifecycle = "optimized"
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        gamma_base = 1 / X_train_scaled.shape[1]
        gamma_min, gamma_max = gamma_base / 100, gamma_base * 10
        print(f"Gamma range: {gamma_min:.2e} - {gamma_max:.2e}")
        optuna_trials = int(S3_CFG.get("optuna_n_trials", 500))
        optuna_n_jobs = int(S3_CFG.get("optuna_n_jobs", 1))
        if optuna_n_jobs == 0:
            optuna_n_jobs = max(1, (os.cpu_count() or 1) - 4)
        optuna_n_jobs = max(1, optuna_n_jobs)
        def objective(trial):
            params = {
                'C': trial.suggest_float('C', 1e-3, 1e3,log=True),
                'gamma': trial.suggest_float('gamma', gamma_min, gamma_max,log=True),
                'kernel': 'rbf',
                'probability': True,
                'random_state': seed
            }
            model = SVC(**params)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            scores = []
            for train_index, valid_index in skf.split(X_train_scaled, y_train):
                X_tr, X_val = X_train_scaled[train_index], X_train_scaled[valid_index]
                y_tr, y_val = y_train[train_index], y_train[valid_index]
                model.fit(X_tr, y_tr)
                probs_val = model.predict_proba(X_val)[:, 1]
                fpr, tpr, _ = metrics.roc_curve(y_val, probs_val)
                scores.append(metrics.auc(fpr, tpr))
            return np.mean(scores)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        study.optimize(objective, 
                    n_trials=optuna_trials,
                    n_jobs=optuna_n_jobs,
                    show_progress_bar=True,
                    )

    best_params = {
        'C': study.best_trial.params['C'],
        'gamma': study.best_trial.params['gamma'],
        'kernel': 'rbf',
        'probability': True,
        'random_state': seed
    }

    model = SVC(**best_params)
    model.fit(X_train_scaled, y_train)

    preds_eval = model.predict(X_eval_scaled)
    probs_eval = model.predict_proba(X_eval_scaled)[:, 1]
    test_acc = metrics.accuracy_score(y_eval, preds_eval)
    conf_matrix_eval = metrics.confusion_matrix(y_eval, preds_eval)
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(y_eval, preds_eval, average='binary')
    specificity = conf_matrix_eval[0, 0] / (conf_matrix_eval[0, 0] + conf_matrix_eval[0, 1])
    fpr_eval, tpr_eval, _ = metrics.roc_curve(y_eval, probs_eval)
    roc_auc_eval = metrics.auc(fpr_eval, tpr_eval)
    metrics_eval = {
        'Accuracy': test_acc,
        'Sensitivity': recall,
        'Specificity': specificity,
        'Precision': precision,
        'F1 Score': f1,
        'AUC': roc_auc_eval
    }

    preds_train = model.predict(X_train_scaled)
    probs_train = model.predict_proba(X_train_scaled)[:, 1]
    train_acc = metrics.accuracy_score(y_train, preds_train)
    conf_matrix_train = metrics.confusion_matrix(y_train, preds_train)
    precision_train, recall_train, f1_train, _ = metrics.precision_recall_fscore_support(y_train, preds_train, average='binary')
    specificity_train = conf_matrix_train[0, 0] / (conf_matrix_train[0, 0] + conf_matrix_train[0, 1])
    fpr_train, tpr_train, _ = metrics.roc_curve(y_train, probs_train)
    roc_auc_train = metrics.auc(fpr_train, tpr_train)
    metrics_train = {
        'Accuracy': train_acc,
        'Sensitivity': recall_train,
        'Specificity': specificity_train,
        'Precision': precision_train,
        'F1 Score': f1_train,
        'AUC': roc_auc_train
    }

    # No latent features available with SVM; return an empty DataFrame for consistency.
    df_bottleneck = pd.DataFrame()

    return (model, df_bottleneck, conf_matrix_train, conf_matrix_eval,
            metrics_train, metrics_eval,
            preds_train, preds_eval, probs_train, probs_eval,
            study, lifecycle)
     
def evaluate_featuregroup(dataset,
                          x_col,y_col,
                          name=None,
                          force_shap_compute=False):
    
    if name is None: name = f"{y_col}-vs-{x_col}"
    
    # ------ Split sets, scaling ------
    X_train = dataset.query("Group == 'normal train' or Group == 'abnormal train'"
                            ).drop(['Group', 'ID'], axis=1)
    y_train = dataset.query("Group == 'normal train' or Group == 'abnormal train'"
                             )['Group'].map({'normal train': 0, 'abnormal train': 1}
                                            ).values
    
    X_eval = dataset.query("Group == 'normal eval' or Group == 'abnormal eval'"
                            ).drop(['Group', 'ID'], axis=1)
    y_eval = dataset.query("Group == 'normal eval' or Group == 'abnormal eval'"
                            )['Group'].map({'normal eval': 0, 'abnormal eval': 1}
                                           ).values
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)
    
    
    # ------ Classification ------
    study_path = os.path.join(STUDIES_DIR, f"study_{name}.pkl")
    study_started_at = _utc_now_iso()
    (clf, df_bottleneck, 
     conf_matrix_train,conf_matrix_eval,
     metrics_train, metrics_eval, 
     preds_train, preds_eval, 
     probs_train, probs_eval,
     study, study_lifecycle
     ) = classifier_SVM_optimized(X_train_scaled, y_train, 
                        X_eval_scaled, y_eval,
                        study_path=study_path,
                        recompute_study=RECOMPUTE_STUDIES,
                        )
    print(f"[{name}] Saving study to {study_path}...")
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)
    _write_study_metadata(
        study_name=name,
        study_path=study_path,
        study_obj=study,
        lifecycle=study_lifecycle,
        started_at=study_started_at,
        ended_at=_utc_now_iso(),
    )
    
    # Figure: performance with ROC, CM, and dca
    fig,axdict = plt.subplot_mosaic([['roc','cm','dca']],
                            figsize=(6, 2.1),
                            num=f"Performance for {name}",)
    fig.set_label(f"Performance for {name}")
    ax_roc,ax_cm,ax_dca= axdict['roc'],axdict['cm'],axdict['dca']

    # ROC curve 
    fpr, tpr, _ = metrics.roc_curve(y_eval, probs_eval)
    roc_auc = metrics.auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'AUC = {roc_auc:.2f}')
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=1.75, linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.0])
    ax_roc.set_xlabel('FPR')
    ax_roc.set_ylabel('TPR')
    #ax_roc.set_title('ROC Curve')
    ax_roc.legend(title=name,loc="lower right", fontsize=9, bbox_to_anchor=(1.02, 0.02))
    ax_roc.set_aspect('equal')

    # Confusion matrix 
    sns.heatmap(conf_matrix_eval, annot=True, fmt='d', cmap='Blues', 
                vmin=0, vmax=np.sum(conf_matrix_eval,axis=(0,1))/2,
                ax=ax_cm, cbar=False, 
                annot_kws={'size': 16},
                xticklabels=['Normal', 'Abnormal'], 
                yticklabels=['Normal', 'Abnormal'])
    #ax_cm.set_title('Confusion Matrix')
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('True')
    ax_cm.set_aspect('equal')

    plt.tight_layout()
    # plt.show((block=False)

    # Decision Curve Analysis
    df_prediction = pd.DataFrame({'Group': y_eval, 
                                    'Prediction': probs_eval})
    df_dca = dca(df_prediction, 
                    outcome='Group', 
                    modelnames=['Prediction'])
    plt.sca(ax_dca)
    plot_graphs(df_dca, 
                graph_type='net_benefit',
                show_grid=False,
                y_limits=(-0.05,0.5))
    ax_dca.set_xlim([0, 1])
    line_pred,line_all,line_none = ax_dca.get_lines()
    line_none.set_linestyle(':')
    line_none.set_linewidth(3)
    line_none.set_color('k')
    line_all.set_color('k')
    line_all.set_linestyle('--')
    line_all.set_linewidth(1.75)
    line_pred.set_color('darkorange')
    line_pred.set_linewidth(2.5)
    line_pred.set_linestyle('-')
    lg = plt.legend(['Prediction','Treat-all','Treat-none'])
    lg.remove()
    
    #ax_dca.set_aspect('equal')
    # plt.show((block=False)
    
    # Convert dictionaries to DataFrames and transpose
    df_metrics_eval = pd.DataFrame(metrics_eval, index=['']).T
    df_metrics_train = pd.DataFrame(metrics_train, index=['']).T
    # Merge metrics DataFrames
    df_metrics = pd.merge(df_metrics_train, df_metrics_eval, 
                          left_index=True, right_index=True, 
                          suffixes=('Train', 'Eval.')
                          )
    # Print metrics
    print(f"\"Metrics Comparison for {name}:\"")
    print(df_metrics.round(3))
    
    return df_metrics

#%% Statistics and classification performance for each feature group

def get_df(fname):
    if fname not in INTERMEDIATE_DF_PATHS:
        raise KeyError(f"Unsupported intermediate dataframe key: {fname}")
    df = pd.read_parquet(INTERMEDIATE_DF_PATHS[fname])
    if fname == "avg_psd" and "average_PSD_dB" not in df.columns and "average_PSD" in df.columns:
        df["average_PSD_dB"] = 10.0 * np.log10(
            np.maximum(df["average_PSD"].to_numpy(dtype=float), np.finfo(float).tiny)
        )
    return df

# HACK: merged dataframe needs merging of map and k columns
df_MSD_merged = get_df("edf_msd")
df_MSD_merged['map-k'] = df_MSD_merged['map'] + df_MSD_merged['k'].astype(str)

tasks = [
    
        # Models on each domain, plus MSD-avg
        
        # PSD
        {'df': get_df("avg_psd"), 
        'x_col': 'f', 'y_col': 'average_PSD_dB',
        'name': 'PSD', 'x_name': 'Frequency [Hz]', 'y_name': r'PSD [$dB \ uV^2/Hz$]',
        },
        # MTMI
        {'df': get_df("edf_mtmi"),
        'x_col': 'k', 'y_col': 'density',
        'name': 'MTMI', 'x_name': r'$t_{MTMI}/\tau$', 'y_name': 'Density',
        },
        # MSD-merged
        {'df': df_MSD_merged,
        'x_col': 'map-k', 'y_col': 'density',
        'name': 'MSD-merged', 'x_name': r'$D(map)/\tau$', 'y_name': 'Density',
        },
        # MSD-avg
        {'df': get_df("avg_msd"),
        'x_col': 'map', 'y_col': 'Duration',
        'name': 'MSD-avg', 'x_name': 'Map', 'y_name': 'Duration [ms]',
        'compress_errorbars_ax': True,
        },
        
        # Models segregating each map
        
        # MSD(A)
        {'df': get_df("edf_msd"
                      ).query("map == 'A'").drop(columns=['map']),
        'x_col': 'k', 'y_col': 'density',
        'name': 'MSD(A)', 'x_name': r'${D(A)}/\tau$', 'y_name': 'Density',
        },
        # MSD(B)
        {'df': get_df("edf_msd"
                      ).query("map == 'B'").drop(columns=['map']),
        'x_col': 'k', 'y_col': 'density',
        'name': 'MSD(B)', 'x_name': r'${D(B)}/\tau$', 'y_name': 'Density',
        },
        # MSD(C)
        {'df': get_df("edf_msd"
                      ).query("map == 'C'").drop(columns=['map']),
        'x_col': 'k', 'y_col': 'density',
        'name': 'MSD(C)', 'x_name': r'${D(C)}/\tau$', 'y_name': 'Density',
        },
        # MSD(D)
        {'df': get_df("edf_msd"
                      ).query("map == 'D'").drop(columns=['map']),
        'x_col': 'k', 'y_col': 'density',
        'name': 'MSD(D)', 'x_name': r'${D(D)}/\tau$', 'y_name': 'Density',
        },
        
        # Models on combinations of domains
        
        # PSD-MTMI
        {'df': [get_df("avg_psd"),
                get_df("edf_mtmi")],
        'x_col': ['f', 'k'], 'y_col': ['average_PSD_dB', 'density'],
        'name': 'PSD-MTMI', 
        'x_name': ['Frequency [Hz]', r'$t_{MTMI}/\tau$'], 
        'y_name': [r'PSD [$dB \ uV^2/Hz$]', 'Density'],
        },
        # MSD-PSD
        {'df': [df_MSD_merged,
                get_df("avg_psd")],
        'x_col': ['map-k', 'f'], 'y_col': ['density', 'average_PSD_dB'],
        'name': 'MSD-PSD', 
        'x_name': [r'$D(map)/\tau$', 'Frequency [Hz]'], 
        'y_name': ['Density', r'PSD [$dB \ uV^2/Hz$]'],
        },
        # MSD-MTMI
        {'df': [df_MSD_merged,
                get_df("edf_mtmi")],
            'x_col': ['map-k', 'k'], 'y_col': ['density', 'density'],
            'name': 'MSD-MTMI', 
            'x_name': [r'$D(map)/\tau$', r'$t_{MTMI}/\tau$'], 
            'y_name': ['Density', 'Density'],
        },
        # MSD-PSD-MTMI
        {'df': [df_MSD_merged,
                get_df("avg_psd"),
                get_df("edf_mtmi")],
            'x_col': ['map-k', 'f', 'k'], 'y_col': ['density', 'average_PSD_dB', 'density'],
            'name': 'MSD-PSD-MTMI', 
            'x_name': [r'$D(map)/\tau$', 'Frequency [Hz]', r'$t_{MTMI}/\tau$'], 
            'y_name': ['Density', r'PSD [$dB \ uV^2/Hz$]', 'Density'],
        }
]


def statistics_and_eval(task):
    # Unpack task
    df = task['df']
    x_col = task['x_col']
    y_col = task['y_col']
    name = task['name']
    x_name = task['x_name']
    y_name = task['y_name']
    compress_errorbars_ax = task.get('compress_errorbars_ax', False)
    
    print(f"\n\nEvaluating {name}...")
    
    # If df is a list, we need concatenation after statistics (but without plot left opened)
    if isinstance(df, list):
        combinations = [f"{x_col[i]}-{y_col[i]}" for i in range(len(x_col))]
        print(f"Detected combination for {name}: {combinations}")
        all_dfs = []
        for _df, _x_col, _y_col, _name, _x_name, _y_name in zip(df, x_col, y_col, name.split('-'), x_name, y_name):
            print("Statistics for ", _name)
            tmp_fig_name = f"tmp-{_name}-for-{name}"
            dataset_sig = statistics_featuregroup(_df,
                                                x_col=_x_col, y_col=_y_col,
                                                name=tmp_fig_name, x_name=_x_name, y_name=_y_name,
                                                compress_errorbars_ax=compress_errorbars_ax)
            dataset_sig.add_prefix(f"{_name}_") # To ensure unique column names
            all_dfs.append(dataset_sig)
            plt.close(tmp_fig_name)
        df = pd.concat(all_dfs, axis=1)
        x_col = 'feature'
        y_col = 'value'
        x_name = f'Feature {name}'
        y_name = 'Value'
    else: 
        # Evaluate feature group
        print("Statistics for ", name)
        dataset_sig = statistics_featuregroup(df,
                                            x_col=x_col, y_col=y_col,
                                            name=name, x_name=x_name, y_name=y_name,
                                            compress_errorbars_ax=compress_errorbars_ax)
    
    # Evaluate classifier performance
    print("Classifier performance for ", name)
    metrics_df = evaluate_featuregroup(dataset_sig,
                                       x_col=x_col, y_col=y_col,
                                       name=name,
                                       force_shap_compute=False)
    
    return metrics_df


task_names = [task['name'] for task in tasks]
results = [statistics_and_eval(task) for task in tasks]
all_metrics = dict(zip(task_names, results))



#%% Table of metrics
# print Table 2 of Results section

# Merge all metrics into a single DataFrame with a multi-index 
# where the first level is the feature group name
merged_metrics = []
for group_name, metrics_df in all_metrics.items():
    metrics_df = metrics_df.copy().round(3).T
    metrics_df.index = pd.MultiIndex.from_product([[group_name], metrics_df.index],
                                                    names=["", ""])
    merged_metrics.append(metrics_df.T)

combined_metrics = pd.concat(merged_metrics, axis=1)
print("Combined Metrics:")
print(combined_metrics)
combined_metrics_path = os.path.join(TABLES_DIR, "metrics_table_side_by_side.csv")
combined_metrics.to_csv(combined_metrics_path)
print(f"[S3] Saved side-by-side metrics table: {combined_metrics_path}")

# # In the paper we print this table column-wise four models at a time
# # so in first row we have models #0, #4, #8, #12 etc. 

# First row
print("Table 2: Performance comparison of different feature groups on the TUAB dataset. \n         Accuracy, Sensitivity, Specificity, Precision, F1 Score, and AUC are \n         reported for both training and evaluation sets.\n")
split_blocks_path = os.path.join(TABLES_DIR, "metrics_table_split_blocks.csv")
with open(split_blocks_path, "w", encoding="utf-8", newline="") as f:
    f.write("table,Table 2 split blocks\n")
    f.write("\n")
    for row in range(4):
        idxs = list(range(0,combined_metrics.shape[1],8))
        idxs.extend(list(range(1,combined_metrics.shape[1],8)))
        idxs.sort()
        idxs = [i+2*row for i in idxs]
        block_df = combined_metrics.iloc[:, idxs]
        print(block_df)
        f.write(f"block,{row + 1}\n")
        block_df.to_csv(f)
        f.write("\n")
print(f"[S3] Saved split metrics table: {split_blocks_path}")

for row in range(4):
    idxs = list(range(0,combined_metrics.shape[1],8))
    idxs.extend(list(range(1,combined_metrics.shape[1],8)))
    idxs.sort()
    idxs = [i+2*row for i in idxs]
    print(combined_metrics.iloc[:,idxs])

# %% Save each active figure with its name
#HACK: ticks of MSD-merged statistics figure
def customtick(ax):
    labels = ax.get_xticklabels()
    newlabels = [lb.get_text()[1:]+'\n'+lb.get_text()[0] for lb in labels]
    ax.set_xticks(ax.get_xticks(), newlabels, fontsize=8)
fig_msd_merged = plt.figure('MSD-merged')
if fig_msd_merged.axes:
    customtick(fig_msd_merged.axes[-1])
# plt.show((block=True)

outfolder = PAPERITEMS_DIR
os.makedirs(outfolder, exist_ok=True)

def _safe_fig_name(fig, idx):
    label = str(fig.get_label() or "").strip()
    if not label:
        label = f"figure_{idx:03d}"
    label = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("._")
    return label or f"figure_{idx:03d}"


for idx, fig_id in enumerate(plt.get_fignums(), start=1):
    fig = plt.figure(fig_id)
    name = _safe_fig_name(fig, idx)
    path_svg = os.path.join(outfolder, f"{name}.svg")
    #path_png = os.path.join(outfolder, f"{name}.png")
    fig.savefig(path_svg, bbox_inches='tight')
    #fig.savefig(path_png, bbox_inches='tight', dpi=300)
    print(f"[S3] Saved figure: {path_svg}")



# %%
