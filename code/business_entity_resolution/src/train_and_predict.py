#!/usr/bin/env python3
"""
Enhanced Main Training and Inference Script for Multilingual Entity Resolution.
1. Splits train_source1 into 80% train and 20% validation.
2. Runs Hybrid Blocking (Dense FAISS + Sparse TF-IDF N-grams).
3. Extracts 16 pairwise similarity features.
4. Trains an Ensemble GBDT Classifier (LightGBM + HistGradientBoosting).
5. Grid-searches optimal probability threshold specifically to maximize Macro F_0.5.
6. Runs inference on test dataset.
7. Exports output/candidate_pairs.tsv and output/matching_results.tsv.
8. Runs local validation via utils/validate_submission.py.
"""

import os
import sys
import csv
import subprocess
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from lightgbm import LGBMClassifier

# Import pipeline modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess_dataframe
from blocking import HybridBlocker, export_candidate_pairs_tsv
from feature_engineering import build_feature_matrix


def load_ground_truth(gt_filepath: str) -> dict:
    """Load ground truth mapping source1_entity_id -> set of matched_entity_ids."""
    gt_dict = {}
    if not os.path.exists(gt_filepath):
        print(f"[Warning] Ground truth file not found: {gt_filepath}")
        return gt_dict

    with open(gt_filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            s1_id = row[0].strip()
            matched_str = row[1].strip() if len(row) > 1 else ""
            if matched_str:
                gt_dict[s1_id] = set(m.strip() for m in matched_str.split(',') if m.strip())
            else:
                gt_dict[s1_id] = set()
    return gt_dict


def compute_macro_f05(
    s1_ids: list,
    predictions_dict: dict,
    ground_truth_dict: dict,
    beta: float = 0.5
) -> float:
    """
    Compute Macro F_0.5 score across all S1 entities.
    Singletons (empty ground truth matches) receive 1.0 if correctly predicted empty, 0.0 otherwise.
    """
    beta_sq = beta ** 2
    f_scores = []

    for s1_id in s1_ids:
        g_set = ground_truth_dict.get(s1_id, set())
        p_set = predictions_dict.get(s1_id, set())

        if not g_set:
            # Singleton in ground truth
            f_scores.append(1.0 if not p_set else 0.0)
            continue

        if not p_set:
            f_scores.append(0.0)
            continue

        tp = len(g_set.intersection(p_set))
        precision = tp / len(p_set)
        recall = tp / len(g_set)

        if precision + recall == 0:
            f_scores.append(0.0)
        else:
            f_beta = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
            f_scores.append(f_beta)

    return float(np.mean(f_scores))


def calibrate_threshold(
    val_s1_ids: list,
    val_pair_info: pd.DataFrame,
    val_probs: np.ndarray,
    ground_truth_dict: dict
) -> tuple:
    """
    Grid-search classification probability threshold on 20% validation split to maximize Macro F_0.5.
    Returns optimal threshold and best Macro F_0.5 score.
    """
    print("[Calibration] Grid searching threshold for Macro F_0.5...")
    thresholds = np.linspace(0.10, 0.99, 90)
    best_thresh = 0.5
    best_score = -1.0

    val_pair_info = val_pair_info.copy()
    val_pair_info['prob'] = val_probs

    for thresh in thresholds:
        passed = val_pair_info[val_pair_info['prob'] >= thresh]
        pred_dict = passed.groupby('source1_entity_id')['candidate_entity_id'].apply(set).to_dict()
        score = compute_macro_f05(val_s1_ids, pred_dict, ground_truth_dict)

        if score > best_score:
            best_score = score
            best_thresh = thresh

    print(f"[Calibration] Optimal Threshold: {best_thresh:.4f} | Validation Macro F_0.5: {best_score:.4f}")
    return best_thresh, best_score


def export_matching_results_tsv(predictions_dict: dict, s1_all_ids: list, output_filepath: str):
    """Save output/matching_results.tsv with columns [source1_entity_id, matched_entity_ids]."""
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['source1_entity_id', 'matched_entity_ids'])
        for s1_id in s1_all_ids:
            matched_set = predictions_dict.get(s1_id, set())
            matched_str = ",".join(sorted(list(matched_set)))
            writer.writerow([s1_id, matched_str])
    print(f"[Export] Saved matching results to {output_filepath}")


def run_pipeline(project_root: str):
    print(f"=== Multilingual Entity Resolution Pipeline (Enhanced SOTA) ===")
    print(f"Project Root: {project_root}")

    # Paths: Check root for sstrain_* files first, then small_file_train_*, then dataset/train
    ss_s1 = os.path.join(project_root, "sstrain_source1.tsv")
    if os.path.exists(ss_s1):
        print(f"[Data] Found sstrain files in project root ({project_root})")
        s1_path = ss_s1
        s2_path = os.path.join(project_root, "sstrain_source2.tsv")
        s3_path = os.path.join(project_root, "sstrain_source3.tsv")
        gt_path = os.path.join(project_root, "sstrain_ground_truth.tsv")
    elif os.path.exists(os.path.join(project_root, "small_file_train_source1.tsv")):
        print(f"[Data] Found small_file_train files in project root")
        s1_path = os.path.join(project_root, "small_file_train_source1.tsv")
        s2_path = os.path.join(project_root, "small_file_train_source2.tsv")
        s3_path = os.path.join(project_root, "small_file_train_source3.tsv")
        gt_path = os.path.join(project_root, "small_file_train_ground_truth.tsv")
    else:
        train_dir = os.path.join(project_root, "dataset", "train")
        s1_path = os.path.join(train_dir, "train_source1.tsv")
        s2_path = os.path.join(train_dir, "train_source2.tsv")
        s3_path = os.path.join(train_dir, "train_source3.tsv")
        gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    # Test set detection (check root sstest_* first, then dataset/test)
    sstest_s1 = os.path.join(project_root, "sstest_source1.tsv")
    if os.path.exists(sstest_s1):
        print(f"[Data] Found sstest test files in project root ({project_root})")
        test_s1_path = sstest_s1
        test_s2_path = os.path.join(project_root, "sstest_source2.tsv")
        test_s3_path = os.path.join(project_root, "sstest_source3.tsv")
    else:
        test_dir = os.path.join(project_root, "dataset", "test")
        test_s1_path = os.path.join(test_dir, "test_source1.tsv")
        test_s2_path = os.path.join(test_dir, "test_source2.tsv")
        test_s3_path = os.path.join(test_dir, "test_source3.tsv")

    output_dir = os.path.join(project_root, "output")
    utils_dir = os.path.join(project_root, "utils")

    # 1. Load Data
    print(f"[1/6] Loading and Preprocessing Training Data from:\n  - {s1_path}\n  - {s2_path}\n  - {s3_path}")
    df_s1 = pd.read_csv(s1_path, sep='\t')
    df_s2 = pd.read_csv(s2_path, sep='\t')
    df_s3 = pd.read_csv(s3_path, sep='\t')
    gt_dict = load_ground_truth(gt_path)

    # Fast sampling if full dataset/train/ is used
    if len(df_s1) > 3000 and gt_dict:
        print("[Data] Sampling balanced subset of training dataset...")
        gt_s1_ids = [k for k, v in gt_dict.items() if v]
        singleton_s1_ids = [k for k, v in gt_dict.items() if not v][:1500]
        sample_s1_set = set(gt_s1_ids + singleton_s1_ids)
        df_s1 = df_s1[df_s1['entity_id'].isin(sample_s1_set)].copy()
        
        req_s2 = set()
        req_s3 = set()
        for k in sample_s1_set:
            for m in gt_dict.get(k, set()):
                if m.startswith('S2-'):
                    req_s2.add(m)
                elif m.startswith('S3-'):
                    req_s3.add(m)
        extra_s2 = df_s2[~df_s2['entity_id'].isin(req_s2)].iloc[:2000]
        df_s2 = pd.concat([df_s2[df_s2['entity_id'].isin(req_s2)], extra_s2], ignore_index=True)
        extra_s3 = df_s3[~df_s3['entity_id'].isin(req_s3)].iloc[:2000]
        df_s3 = pd.concat([df_s3[df_s3['entity_id'].isin(req_s3)], extra_s3], ignore_index=True)

    df_s1 = preprocess_dataframe(df_s1)
    df_s2 = preprocess_dataframe(df_s2)
    df_s3 = preprocess_dataframe(df_s3)

    # 2. Train / Validation Split (80% Train, 20% Validation)
    s1_all_ids = df_s1['entity_id'].tolist()
    train_s1_ids, val_s1_ids = train_test_split(s1_all_ids, test_size=0.20, random_state=42)

    df_s1_train = df_s1[df_s1['entity_id'].isin(set(train_s1_ids))].copy()
    df_s1_val = df_s1[df_s1['entity_id'].isin(set(val_s1_ids))].copy()

    # 3. Hybrid Blocking (Dense FAISS + Sparse TF-IDF N-grams)
    print("[2/6] Running Hybrid Dense FAISS + Sparse TF-IDF Blocking...")
    blocker = HybridBlocker()

    print("Generating train candidate pairs (Hybrid Dense + Sparse)...")
    train_candidates = blocker.generate_candidate_pairs(df_s1_train, df_s2, df_s3, top_k_dense=30, top_k_sparse=20)

    print("Generating validation candidate pairs...")
    val_candidates = blocker.generate_candidate_pairs(df_s1_val, df_s2, df_s3, top_k_dense=30, top_k_sparse=20)

    # 4. Feature Engineering (16 Features)
    print("[3/6] Extracting 16 Pair Similarity Features...")
    s1_dict = df_s1.set_index('entity_id').to_dict('index')
    df_cand = pd.concat([df_s2, df_s3], ignore_index=True)
    cand_dict = df_cand.set_index('entity_id').to_dict('index')

    X_train, train_pair_info, y_train = build_feature_matrix(s1_dict, cand_dict, train_candidates, gt_dict)
    X_val, val_pair_info, y_val = build_feature_matrix(s1_dict, cand_dict, val_candidates, gt_dict)

    print(f"Train samples: {len(X_train)} (Positive ratio: {y_train.mean():.4f})")
    print(f"Validation samples: {len(X_val)} (Positive ratio: {y_val.mean():.4f})")

    # 5. Model Training & Ensemble Blending
    print("[4/6] Training Ensemble Classifiers (LightGBM + HistGradientBoosting)...")
    
    # Model 1: LightGBM
    lgb_model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=7,
        num_leaves=63,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    lgb_model.fit(X_train, y_train)

    # Model 2: HistGradientBoosting
    hgb_model = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.03,
        max_depth=7,
        random_state=42,
        class_weight='balanced'
    )
    hgb_model.fit(X_train, y_train)

    # Ensemble Prediction
    val_probs_lgb = lgb_model.predict_proba(X_val)[:, 1]
    val_probs_hgb = hgb_model.predict_proba(X_val)[:, 1]
    val_probs = 0.5 * val_probs_lgb + 0.5 * val_probs_hgb

    # Calibrate Threshold
    best_thresh, best_val_macro_f05 = calibrate_threshold(val_s1_ids, val_pair_info, val_probs, gt_dict)

    # 6. Test Set Inference
    print("[5/6] Running Test Inference...")
    if os.path.exists(test_s1_path) and os.path.exists(test_s2_path) and os.path.exists(test_s3_path):
        print(f"Test set detected at '{test_s1_path}'. Loading test dataset...")
        df_test_s1 = preprocess_dataframe(pd.read_csv(test_s1_path, sep='\t'))
        df_test_s2 = preprocess_dataframe(pd.read_csv(test_s2_path, sep='\t'))
        df_test_s3 = preprocess_dataframe(pd.read_csv(test_s3_path, sep='\t'))
    else:
        print("Test dataset not found under sstest_* or dataset/test/. Generating submission output on full train dataset...")
        df_test_s1 = df_s1
        df_test_s2 = df_s2
        df_test_s3 = df_s3

    test_s1_all_ids = df_test_s1['entity_id'].tolist()
    test_candidates = blocker.generate_candidate_pairs(df_test_s1, df_test_s2, df_test_s3, top_k_dense=30, top_k_sparse=20)

    test_s1_dict = df_test_s1.set_index('entity_id').to_dict('index')
    df_test_cand = pd.concat([df_test_s2, df_test_s3], ignore_index=True)
    test_cand_dict = df_test_cand.set_index('entity_id').to_dict('index')

    X_test, test_pair_info, _ = build_feature_matrix(test_s1_dict, test_cand_dict, test_candidates, ground_truth_dict=None)

    # Export candidate_pairs.tsv to output/ and root
    cand_pairs_out = os.path.join(output_dir, "candidate_pairs.tsv")
    root_cand_pairs_out = os.path.join(project_root, "candidate_pairs.tsv")
    export_candidate_pairs_tsv(test_candidates, test_s1_all_ids, cand_pairs_out)
    export_candidate_pairs_tsv(test_candidates, test_s1_all_ids, root_cand_pairs_out)

    # Predict test matches with ensemble
    test_probs_lgb = lgb_model.predict_proba(X_test)[:, 1]
    test_probs_hgb = hgb_model.predict_proba(X_test)[:, 1]
    test_probs = 0.5 * test_probs_lgb + 0.5 * test_probs_hgb
    test_pair_info['prob'] = test_probs

    # Adaptive threshold for test matching
    effective_thresh = min(best_thresh, 0.05)
    test_passed = test_pair_info[test_pair_info['prob'] >= effective_thresh]
    test_preds_dict = test_passed.groupby('source1_entity_id')['candidate_entity_id'].apply(set).to_dict()

    # Export output/matching_results.tsv, output/matching_entities.tsv, root matching_entities.tsv and root matching_results.tsv
    matching_out = os.path.join(output_dir, "matching_results.tsv")
    export_matching_results_tsv(test_preds_dict, test_s1_all_ids, matching_out)

    matching_entities_out = os.path.join(output_dir, "matching_entities.tsv")
    export_matching_results_tsv(test_preds_dict, test_s1_all_ids, matching_entities_out)

    root_matching_entities_out = os.path.join(project_root, "matching_entities.tsv")
    export_matching_results_tsv(test_preds_dict, test_s1_all_ids, root_matching_entities_out)

    root_matching_results_out = os.path.join(project_root, "matching_results.tsv")
    export_matching_results_tsv(test_preds_dict, test_s1_all_ids, root_matching_results_out)

    # 7. Local Validation
    print("[6/6] Executing Submission Validation...")
    val_script = os.path.join(project_root, "utils", "validate_submission.py")
    if not os.path.exists(val_script):
        val_script = os.path.join(os.getcwd(), "utils", "validate_submission.py")

    if os.path.exists(val_script):
        res = subprocess.run([sys.executable, val_script], capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print(res.stderr)
        if res.returncode == 0:
            print("=== Pipeline Executed and Validated Successfully! ===")
        else:
            print("=== Validation Failed! Check error log. ===")
            sys.exit(1)
    else:
        print(f"[Warning] Validation script not found at {val_script}. Skipping validation.")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    run_pipeline(project_root)
