# -*- coding: utf-8 -*-
"""
GMM-UBM Speaker Verification
----------------------------
Trains a Universal Background Model (UBM) on pooled multi-speaker MFCCs,
then MAP-adapts an owner-specific GMM for two-factor voice authentication
(PIN digits + speaker identity).
"""
import copy
import os
import numpy as np
from sklearn.mixture import GaussianMixture
from utils import mfcc


def collect_speaker_features(records_root, cmvn=False):
    """
    Collect MFCC feature matrices grouped by speaker name from `records_root`.
    Supports either:
      - records_root/<speaker>/digit_<d>/*.wav
      - records_root/digit_<d>/*.wav  (mapped to speaker 'default')
    """
    speakers = {}
    if not os.path.isdir(records_root):
        return speakers

    entries = sorted(os.listdir(records_root))
    has_digit_dirs = any(e.startswith("digit_") for e in entries)
    if has_digit_dirs:
        feats = []
        for d in range(10):
            ddir = os.path.join(records_root, f"digit_{d}")
            if not os.path.isdir(ddir):
                continue
            for f in sorted(os.listdir(ddir)):
                if f.lower().endswith(".wav"):
                    feats.append(mfcc(os.path.join(ddir, f), cmvn=cmvn))
        if feats:
            speakers["default"] = feats
        return speakers

    for spk in entries:
        spk_dir = os.path.join(records_root, spk)
        if not os.path.isdir(spk_dir):
            continue
        feats = []
        for d in range(10):
            ddir = os.path.join(spk_dir, f"digit_{d}")
            if not os.path.isdir(ddir):
                continue
            for f in sorted(os.listdir(ddir)):
                if f.lower().endswith(".wav"):
                    feats.append(mfcc(os.path.join(ddir, f), cmvn=cmvn))
        if feats:
            speakers[spk] = feats
    return speakers


def map_adapt_ubm(ubm, owner_X, relevance_factor=16.0):
    """MAP-adapt UBM component means using owner's pooled MFCC frames."""
    adapted = copy.deepcopy(ubm)
    resp = ubm.predict_proba(owner_X)
    n_k = np.maximum(resp.sum(axis=0), 1e-12)
    f_k = resp.T @ owner_X
    alpha = (n_k / (n_k + relevance_factor))[:, None]
    new_means = alpha * (f_k / n_k[:, None]) + (1.0 - alpha) * ubm.means_
    adapted.means_ = new_means
    return adapted


def train_speaker_verifier(owner_name, my_records_dir="./my_records", background_dir="./processed_train_records", n_components=24, cmvn=True):
    """
    Train UBM + MAP-adapted owner GMM (always uses CMVN by default to remove microphone/channel bias).
    Returns a dict with {'owner': owner_name, 'ubm': ubm, 'owner_gmm': owner_gmm, 'threshold': float, 'cmvn': bool}
    or None if no recordings exist for `owner_name`.
    """
    enrolled = collect_speaker_features(my_records_dir, cmvn=cmvn)
    if owner_name not in enrolled or len(enrolled[owner_name]) == 0:
        return None

    owner_seqs = enrolled[owner_name]
    owner_X = np.vstack(owner_seqs)

    bg_seqs = []
    bg_map = collect_speaker_features(background_dir, cmvn=cmvn)
    for seqs in bg_map.values():
        bg_seqs.extend(seqs)
    for spk, seqs in enrolled.items():
        if spk != owner_name:
            bg_seqs.extend(seqs)

    ubm_seqs = bg_seqs if bg_seqs else owner_seqs
    ubm_X = np.vstack(ubm_seqs)

    ubm = GaussianMixture(n_components=n_components, covariance_type='diag', max_iter=100, random_state=0)
    np.seterr(all='ignore')
    ubm.fit(ubm_X)

    owner_gmm = map_adapt_ubm(ubm, owner_X, relevance_factor=8.0)

    # Calibrate LLR threshold using owner vs background utterances
    owner_llrs = [float(owner_gmm.score(s) - ubm.score(s)) for s in owner_seqs]
    if bg_seqs:
        bg_llrs = [float(owner_gmm.score(s) - ubm.score(s)) for s in bg_seqs[:60]]
        threshold = 0.5 * (float(np.percentile(owner_llrs, 20)) + float(np.percentile(bg_llrs, 90)))
    else:
        threshold = float(np.percentile(owner_llrs, 15)) - 0.3

    return {
        "owner": owner_name,
        "ubm": ubm,
        "owner_gmm": owner_gmm,
        "threshold": float(threshold),
        "cmvn": bool(cmvn),
    }


def verify_speaker(verifier, features_mfcc):
    """
    Compute log-likelihood ratio (LLR) for an utterance.
    :return: (is_owner_bool, llr_float, threshold_float)
    """
    if verifier is None:
        return True, 0.0, 0.0
    llr = float(verifier["owner_gmm"].score(features_mfcc) - verifier["ubm"].score(features_mfcc))
    threshold = float(verifier["threshold"])
    return (llr >= threshold), llr, threshold
