"""Calibration diagnostics for persisted NLI prediction records."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import brier_score_loss

def analyze_confidence(prediction_paths, bins=10):
    rows=[]
    for path in prediction_paths:
        rows += [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x]
    usable=[r for r in rows if r.get('confidence') is not None]
    if not usable: return {"count":0,"warning":"Persisted predictions do not include confidence."}
    confidence=np.array([float(r['confidence']) for r in usable]); correct=np.array([r['true_label']==r['predicted_label'] for r in usable],dtype=int)
    ece=0.0
    for lo,hi in zip(np.linspace(0,1,bins,endpoint=False),np.linspace(1/bins,1,bins)):
        mask=(confidence>=lo)&((confidence<hi) if hi<1 else (confidence<=hi))
        if mask.any(): ece += mask.mean()*abs(confidence[mask].mean()-correct[mask].mean())
    return {"count":len(usable),"correct_mean_confidence":float(confidence[correct==1].mean()) if correct.any() else None,"incorrect_mean_confidence":float(confidence[correct==0].mean()) if (~correct.astype(bool)).any() else None,"brier_score":float(brier_score_loss(correct,confidence)),"expected_calibration_error":float(ece)}
