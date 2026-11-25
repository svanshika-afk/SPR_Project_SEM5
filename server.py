# -*- coding: utf-8 -*-
"""
FastAPI Backend for Voice-PIN Security Demo
-------------------------------------------
Serves the Web UI and provides endpoints for:
- /api/status  : model availability, enrolled speakers & sample counts
- /api/enroll  : auto-save browser-recorded digit clips (with VAD trim & 80/20 split)
- /api/train   : retrain all digit models + score thresholds + GMM-UBM speaker verifier
- /api/predict : classify spoken digit + OOV/margin rejection + speaker verification
"""
import io
import os
import shutil
from typing import Optional
import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import dtw
import gmm
import gmm_from_sklearn
import hmm
import hmm_from_hmmlearn
from speaker_id import verify_speaker
from train_all import train_all_models
from utils import load_models, mfcc_from_array, trim_silence


SAMPLE_RATE = 22050
MIN_CLIP_SEC = 0.18
MAX_CLIP_SEC = 1.25

MODEL_SPECS = {
    "HMM (hmmlearn)": ("./models/hmm_hmmlearn.joblib", hmm_from_hmmlearn.predict_from_features, 4.0),
    "GMM (custom)": ("./models/gmm_custom.joblib", gmm.predict_from_features, 4.0),
    "GMM (sklearn)": ("./models/gmm_sklearn.joblib", gmm_from_sklearn.predict_from_features, 0.04),
    "HMM (custom)": ("./models/hmm_custom.joblib", hmm.predict_from_features, 4.0),
    "DTW (multi-template)": ("./models/dtw_templates.joblib", dtw.predict_from_features, 0.20),
}

app = FastAPI(title="SPR Voice-PIN Demo")
os.makedirs("./static", exist_ok=True)
app.mount("/static", StaticFiles(directory="./static"), name="static")

state = {
    "models": {},
    "thresholds": {},
    "speaker_verifier": None,
}


def reload_models():
    """Load all available trained models, score thresholds, and speaker verifier from ./models/."""
    loaded = {}
    for name, (path, _, _) in MODEL_SPECS.items():
        if os.path.isfile(path):
            try:
                loaded[name] = load_models(path)
            except Exception:
                pass
    state["models"] = loaded

    thresh_path = "./models/score_thresholds.joblib"
    state["thresholds"] = load_models(thresh_path) if os.path.isfile(thresh_path) else {}

    spk_path = "./models/speaker_verifier.joblib"
    state["speaker_verifier"] = load_models(spk_path) if os.path.isfile(spk_path) else None


reload_models()


def _decode_wav(raw_bytes: bytes):
    """Read WAV bytes and resample to 22050 Hz mono float32."""
    y, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    return y.astype(np.float32)


def _get_enrolled_counts():
    """Return {speaker: {digit_str: count}} from ./my_records."""
    enrolled = {}
    root = "./my_records"
    if not os.path.isdir(root):
        return enrolled
    for spk in sorted(os.listdir(root)):
        spk_dir = os.path.join(root, spk)
        if not os.path.isdir(spk_dir) or spk.startswith("digit_"):
            continue
        counts = {}
        for d in range(10):
            ddir = os.path.join(spk_dir, f"digit_{d}")
            counts[str(d)] = len([f for f in os.listdir(ddir) if f.lower().endswith(".wav")]) if os.path.isdir(ddir) else 0
        enrolled[spk] = counts
    return enrolled


@app.get("/")
def index():
    return FileResponse("./static/index.html")


@app.get("/api/status")
def get_status():
    verifier = state["speaker_verifier"]
    return {
        "models": list(state["models"].keys()),
        "speaker_owner": verifier["owner"] if verifier else None,
        "enrolled": _get_enrolled_counts(),
    }


@app.post("/api/enroll")
async def enroll_clip(
    file: UploadFile = File(...),
    digit: int = Form(...),
    speaker: str = Form("owner"),
):
    if digit < 0 or digit > 9:
        raise HTTPException(status_code=400, detail="Digit must be between 0 and 9.")
    speaker_clean = "".join(c for c in speaker.strip() if c.isalnum() or c in ("_", "-")) or "owner"

    raw_bytes = await file.read()
    try:
        y = _decode_wav(raw_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid audio: {exc}")

    trimmed, is_valid = trim_silence(y, SAMPLE_RATE, top_db=25, min_duration=0.12)
    dur = len(trimmed) / float(SAMPLE_RATE)
    raw_peak = float(np.max(np.abs(y))) if y.size else 0.0
    if not is_valid:
        return {
            "saved": False,
            "reason": f"Too quiet or too short (peak={raw_peak:.4f}, trimmed={dur:.2f}s) — check mic selection or speak louder.",
        }
    if dur > MAX_CLIP_SEC:
        return {"saved": False, "reason": f"Clip too long ({dur:.2f}s) — speak only one digit."}

    raw_dir = os.path.join("./my_records", speaker_clean, f"digit_{digit}")
    train_dir = os.path.join("./my_train_records", f"digit_{digit}")
    test_dir = os.path.join("./my_test_records", f"digit_{digit}")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    existing_raw = len([f for f in os.listdir(raw_dir) if f.lower().endswith(".wav")])
    idx = existing_raw + 1
    sf.write(os.path.join(raw_dir, f"{idx}_{digit}.wav"), y, SAMPLE_RATE)

    # Global index across speakers so train/test stay balanced 80/20
    total_train = len([f for f in os.listdir(train_dir) if f.lower().endswith(".wav")])
    total_test = len([f for f in os.listdir(test_dir) if f.lower().endswith(".wav")])
    total = total_train + total_test + 1
    split = "test" if (total % 5 == 0) else "train"
    target_dir = test_dir if split == "test" else train_dir
    sf.write(os.path.join(target_dir, f"{speaker_clean}_{idx}_{digit}.wav"), trimmed, SAMPLE_RATE)

    return {
        "saved": True,
        "speaker": speaker_clean,
        "digit": digit,
        "split": split,
        "duration_sec": round(dur, 2),
        "enrolled": _get_enrolled_counts(),
    }


@app.delete("/api/enroll/{speaker}")
def delete_speaker(speaker: str):
    speaker_clean = "".join(c for c in speaker.strip() if c.isalnum() or c in ("_", "-"))
    spk_dir = os.path.join("./my_records", speaker_clean)
    if os.path.isdir(spk_dir):
        shutil.rmtree(spk_dir)
    # Remove speaker's files from my_train_records / my_test_records
    for root_dir in ("./my_train_records", "./my_test_records"):
        if not os.path.isdir(root_dir):
            continue
        for d in range(10):
            ddir = os.path.join(root_dir, f"digit_{d}")
            if os.path.isdir(ddir):
                for f in os.listdir(ddir):
                    if f.startswith(f"{speaker_clean}_"):
                        os.remove(os.path.join(ddir, f))
    return {"deleted": speaker_clean, "enrolled": _get_enrolled_counts()}


class TrainRequest(BaseModel):
    cmvn: bool = False
    owner: Optional[str] = None


@app.post("/api/train")
def train_endpoint(req: TrainRequest):
    summary = train_all_models(cmvn=req.cmvn, owner=req.owner or None)
    reload_models()
    return {
        **summary,
        "models": list(state["models"].keys()),
    }


@app.post("/api/predict")
async def predict_clip(
    file: UploadFile = File(...),
    model: str = Form("HMM (hmmlearn)"),
    reject_unclear: bool = Form(True),
    verify_spk: bool = Form(False),
):
    if model not in state["models"]:
        raise HTTPException(status_code=400, detail=f"Model '{model}' is not loaded.")

    raw_bytes = await file.read()
    try:
        y = _decode_wav(raw_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid audio: {exc}")

    trimmed, is_valid = trim_silence(y, SAMPLE_RATE, top_db=25, min_duration=MIN_CLIP_SEC)
    dur = len(trimmed) / float(SAMPLE_RATE)
    if not is_valid:
        return {"accepted": False, "reason": "Ignored short or quiet sound.", "duration_sec": round(dur, 2)}
    if dur > MAX_CLIP_SEC:
        return {"accepted": False, "reason": f"Too long ({dur:.2f}s) — please pause between digits.", "duration_sec": round(dur, 2)}

    _, predict_fn, min_margin = MODEL_SPECS[model]
    feats = mfcc_from_array(trimmed, SAMPLE_RATE, delta=2, cmvn=False)
    digit, margin, scores = predict_fn(state["models"][model], feats)

    # Rejection gate 1: margin between top-1 and top-2
    if reject_unclear and margin < min_margin:
        return {
            "accepted": False,
            "reason": f"Unclear speech (closest: {digit}, margin={margin:.2f} < {min_margin:.2f}).",
            "digit": digit,
            "margin": round(margin, 2),
            "duration_sec": round(dur, 2),
        }

    # Rejection gate 2: per-frame score threshold against training distribution (catches OOV like 'banana')
    model_thresholds = state["thresholds"].get(model, {})
    if reject_unclear and digit in model_thresholds:
        raw_best = float(scores[digit])
        norm_score = raw_best if model.startswith("DTW") else raw_best / max(len(feats), 1)
        if norm_score < model_thresholds[digit]:
            return {
                "accepted": False,
                "reason": f"Out-of-vocabulary / unclear (score={norm_score:.2f} < {model_thresholds[digit]:.2f}).",
                "digit": digit,
                "margin": round(margin, 2),
                "duration_sec": round(dur, 2),
            }

    # Optional two-factor GMM-UBM speaker verification
    spk_ok, spk_llr, spk_thresh = True, 0.0, 0.0
    if verify_spk and state["speaker_verifier"] is not None:
        spk_ok, spk_llr, spk_thresh = verify_speaker(state["speaker_verifier"], feats)

    return {
        "accepted": True,
        "digit": digit,
        "margin": round(margin, 2),
        "duration_sec": round(dur, 2),
        "speaker_ok": spk_ok,
        "speaker_llr": round(spk_llr, 3),
        "speaker_threshold": round(spk_thresh, 3),
    }
