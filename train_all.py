# -*- coding: utf-8 -*-
"""Train and evaluate all digit recognition models (+ optional GMM-UBM speaker verifier)."""
import argparse
import os
import numpy as np
import dtw
import gmm
import gmm_from_sklearn
import hmm
import hmm_from_hmmlearn
from speaker_id import train_speaker_verifier
from utils import get_mfcc_sequences, get_test_files, mfcc, save_models


def compute_score_thresholds(models, predict_fn, train_dir, cmvn=False, percentile=5):
    """
    Compute a per-digit minimum per-frame score threshold from training utterances
    so out-of-vocabulary words (e.g., 'banana') can be rejected.
    """
    sequences = get_mfcc_sequences(train_dir, cmvn=cmvn)
    thresholds = {}
    for digit, seqs in sequences.items():
        if not seqs:
            continue
        per_frame_scores = []
        for s in seqs:
            pred, _, scores = predict_fn(models, s)
            # Normalize by frame count (DTW is already length-normalized)
            best_score = float(scores[digit])
            norm_score = best_score if predict_fn is dtw.predict_from_features else best_score / max(len(s), 1)
            per_frame_scores.append(norm_score)
        thresholds[digit] = float(np.percentile(per_frame_scores, percentile)) - 1.0
    return thresholds


def evaluate(name, models, predict_fn, test_dir, cmvn):
    test_items = get_test_files(test_dir)
    if not test_items:
        print(f"[{name}] No test files found in {test_dir}; skipping evaluation.")
        return None
    correct = 0
    for path, truth in test_items:
        feats = mfcc(path, cmvn=cmvn)
        pred, _, _ = predict_fn(models, feats)
        if pred == truth:
            correct += 1
    acc = correct / len(test_items)
    print(f"[{name}] Test accuracy: {correct}/{len(test_items)} ({acc:.2%})")
    return {
        "model": name,
        "correct": correct,
        "total": len(test_items),
        "accuracy": round(acc, 4),
    }


def train_all_models(train_dir=None, test_dir=None, cmvn=False, owner=None):
    """Programmatic entry point used by both CLI and FastAPI /api/train."""
    if train_dir is None:
        train_dir = "./my_train_records" if os.path.isdir("./my_train_records") else "./processed_train_records"
    if test_dir is None:
        test_dir = "./my_test_records" if os.path.isdir("./my_test_records") else "./processed_test_records"

    print(f"Training on: {train_dir}")
    print(f"Testing on:  {test_dir}")
    print(f"CMVN:        {cmvn}\n")

    results = []
    thresholds = {}

    dtw_model = dtw.train_model_dtw(train_dir, cmvn=cmvn)
    save_models(dtw_model, "./models/dtw_templates.joblib")
    thresholds["DTW (multi-template)"] = compute_score_thresholds(dtw_model, dtw.predict_from_features, train_dir, cmvn)
    results.append(evaluate("DTW (multi-template)", dtw_model, dtw.predict_from_features, test_dir, cmvn))

    gmm_sk = gmm_from_sklearn.train_model_gmm(train_dir, cmvn=cmvn)
    save_models(gmm_sk, "./models/gmm_sklearn.joblib")
    thresholds["GMM (sklearn)"] = compute_score_thresholds(gmm_sk, gmm_from_sklearn.predict_from_features, train_dir, cmvn)
    results.append(evaluate("GMM (sklearn)", gmm_sk, gmm_from_sklearn.predict_from_features, test_dir, cmvn))

    hmm_hl = hmm_from_hmmlearn.train_model_hmm(train_dir, cmvn=cmvn)
    save_models(hmm_hl, "./models/hmm_hmmlearn.joblib")
    thresholds["HMM (hmmlearn)"] = compute_score_thresholds(hmm_hl, hmm_from_hmmlearn.predict_from_features, train_dir, cmvn)
    results.append(evaluate("HMM (hmmlearn)", hmm_hl, hmm_from_hmmlearn.predict_from_features, test_dir, cmvn))

    gmm_custom = gmm.train_model_gmm(train_dir, n_components=6, cmvn=cmvn)
    save_models(gmm_custom, "./models/gmm_custom.joblib")
    thresholds["GMM (custom)"] = compute_score_thresholds(gmm_custom, gmm.predict_from_features, train_dir, cmvn)
    results.append(evaluate("GMM (custom)", gmm_custom, gmm.predict_from_features, test_dir, cmvn))

    hmm_custom = hmm.train_model_hmm(train_dir, n_components=3, max_iter=100, cmvn=cmvn)
    save_models(hmm_custom, "./models/hmm_custom.joblib")
    thresholds["HMM (custom)"] = compute_score_thresholds(hmm_custom, hmm.predict_from_features, train_dir, cmvn)
    results.append(evaluate("HMM (custom)", hmm_custom, hmm.predict_from_features, test_dir, cmvn))

    save_models(thresholds, "./models/score_thresholds.joblib")
    save_models({"cmvn": cmvn}, "./models/train_config.joblib")

    speaker_trained = None
    if owner:
        verifier = train_speaker_verifier(owner, cmvn=True)
        if verifier is not None:
            save_models(verifier, "./models/speaker_verifier.joblib")
            speaker_trained = owner
            print(f"[Speaker Verification] Trained GMM-UBM for owner '{owner}' (threshold={verifier['threshold']:.3f})")

    print("\nAll models saved to ./models/")
    return {
        "train_dir": train_dir,
        "test_dir": test_dir,
        "metrics": [r for r in results if r is not None],
        "speaker_owner": speaker_trained,
    }


def main():
    parser = argparse.ArgumentParser(description="Train all models and save them to ./models/")
    parser.add_argument("--train-dir", default=None, help="Training directory")
    parser.add_argument("--test-dir", default=None, help="Test directory")
    parser.add_argument("--cmvn", action="store_true", help="Enable per-utterance CMVN normalization")
    parser.add_argument("--owner", default=None, help="Owner speaker name in ./my_records/ for GMM-UBM verification")
    args = parser.parse_args()
    train_all_models(train_dir=args.train_dir, test_dir=args.test_dir, cmvn=args.cmvn, owner=args.owner)


if __name__ == "__main__":
    main()
