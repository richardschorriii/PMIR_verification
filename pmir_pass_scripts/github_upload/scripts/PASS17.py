#!/usr/bin/env python3
"""
PASS17 - PMIR Verification Test
Extracted from ChatGPT transcript (line 60771)
Length: 1120 characters
"""

def best_bal_threshold(y_true, p, grid=401):
    # maximize balanced accuracy over thresholds in [0,1]
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    ts = np.linspace(0.0, 1.0, int(grid))
    best_t = 0.5
    best_b = -1.0
    for t in ts:
        yhat = (p >= t).astype(int)
        b = balanced_acc(y_true, yhat)
        if b > best_b:
            best_b = b
            best_t = float(t)
    return best_t, float(best_b)

# ---- choose t* on CAL and evaluate on TEST ----
tstar_raw, _ = best_bal_threshold(y_cal, sigmoid(z_cal), grid=401)
tstar_platt, _ = best_bal_threshold(y_cal, sigmoid(a * z_cal0 + b), grid=401)

yhat_raw_tstar = (p_raw >= tstar_raw).astype(int)
yhat_platt_tstar = (p_platt >= tstar_platt).astype(int)

bal_raw_tstar = balanced_acc(y_test, yhat_raw_tstar)
bal_platt_tstar = balanced_acc(y_test, yhat_platt_tstar)
Then add these fields into the out_rows.append({...}) dict:

"tstar_raw": tstar_raw,
"tstar_platt": tstar_platt,
"bal_raw_tstar": bal_raw_tstar,
"bal_platt_tstar": bal_platt_tstar,
And update summarize() to print them (optional but recommended).