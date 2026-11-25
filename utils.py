# -*- coding: utf-8 -*-
import os
import warnings
import joblib
import numpy as np
import librosa


def mfcc_from_array(y, sr, delta=2, cmvn=True):
    """
    Calculate MFCC features (plus optional delta and delta-delta) from an audio array.
    :param y: 1D audio signal array
    :param sr: sample rate of y
    :param delta: derivative order (0, 1, or 2), default is 2
    :param cmvn: apply utterance-level Cepstral Mean and Variance Normalization
    :return: (T, D) feature matrix
    """
    y = np.asarray(y, dtype=np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1e-6:
        y = y * (0.9 / peak)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mfcc_feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        ans = [mfcc_feat]
        if delta >= 1 and mfcc_feat.shape[1] >= 2:
            width = min(9, mfcc_feat.shape[1])
            if width % 2 == 0:
                width -= 1
            if width >= 3:
                mfcc_delta1 = librosa.feature.delta(mfcc_feat, width=width, order=1, mode='nearest')
                ans.append(mfcc_delta1)
                if delta >= 2:
                    mfcc_delta2 = librosa.feature.delta(mfcc_feat, width=width, order=2, mode='nearest')
                    ans.append(mfcc_delta2)
            else:
                if delta >= 1:
                    ans.append(np.zeros_like(mfcc_feat))
                if delta >= 2:
                    ans.append(np.zeros_like(mfcc_feat))
        elif delta >= 1:
            ans.append(np.zeros_like(mfcc_feat))
            if delta >= 2:
                ans.append(np.zeros_like(mfcc_feat))

    feats = np.transpose(np.concatenate(ans, axis=0), [1, 0])
    if cmvn and feats.shape[0] > 1:
        mean = np.mean(feats, axis=0, keepdims=True)
        std = np.std(feats, axis=0, keepdims=True)
        feats = (feats - mean) / np.maximum(std, 1e-8)
    return feats


def mfcc(wav_path, delta=2, cmvn=True):
    """
    Read a .wav file and calculate MFCC features.
    :param wav_path: path of audio file
    :param delta: derivative order, default order is 2
    :param cmvn: apply utterance-level CMVN
    :return: (T, D) feature matrix
    """
    y, sr = librosa.load(wav_path, sr=22050)
    return mfcc_from_array(y, sr, delta=delta, cmvn=cmvn)


def get_mfcc_sequences(data_dir, delta=2, cmvn=True):
    """
    Load MFCC sequences for digits 0..9 in deterministic sorted order.
    :param data_dir: directory containing digit_0 .. digit_9 folders
    :return: dict mapping int digit (0..9) -> list of (T, D) MFCC arrays
    """
    sequences = {}
    for i in range(10):
        digit_dir = os.path.join(data_dir, f"digit_{i}")
        if not os.path.isdir(digit_dir):
            continue
        wav_files = sorted(f for f in os.listdir(digit_dir) if f.lower().endswith('.wav'))
        sequences[i] = [
            mfcc(os.path.join(digit_dir, f), delta=delta, cmvn=cmvn)
            for f in wav_files
        ]
    return sequences


def get_test_files(test_dir):
    """
    Return sorted list of (file_path, ground_truth_digit_int) from test_dir.
    """
    items = []
    for i in range(10):
        digit_dir = os.path.join(test_dir, f"digit_{i}")
        if not os.path.isdir(digit_dir):
            continue
        for f in sorted(os.listdir(digit_dir)):
            if f.lower().endswith('.wav'):
                items.append((os.path.join(digit_dir, f), i))
    return items


def trim_silence(y, sr, top_db=25, min_duration=0.12):
    """
    Trim leading and trailing silence from a 1D audio array.
    :return: (trimmed_y, is_valid_speech)
    """
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return y, False
    peak = float(np.max(np.abs(y)))
    if peak < 1e-4:
        return y, False
    trimmed, _ = librosa.effects.trim(y, top_db=top_db, frame_length=512, hop_length=128)
    is_valid = (len(trimmed) / float(sr)) >= min_duration and float(np.max(np.abs(trimmed))) >= 1e-4
    return trimmed, is_valid


def save_models(models, path):
    """Save trained models to disk with joblib."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(models, path)


def load_models(path):
    """Load trained models from disk with joblib."""
    return joblib.load(path)
