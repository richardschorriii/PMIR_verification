#!/usr/bin/env python3
"""
PASS16 - PMIR Verification Test
Extracted from ChatGPT transcript (line 60338)
Length: 1913 characters
"""

def best_threshold_balanced_acc(y, p, grid=401):
    """
    Choose threshold t in [0,1] maximizing balanced accuracy on (y,p).
    Returns (t_best, bal_best, acc_at_best).
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)

    # protect edge cases
    if len(np.unique(y)) < 2:
        return 0.5, np.nan, np.nan

    ts = np.linspace(0.0, 1.0, int(grid))
    best = (-1.0, 0.5, np.nan, np.nan)  # (bal, t, acc, bal)
    best_bal = -1.0
    best_t = 0.5
    best_acc = np.nan

    for t in ts:
        yhat = (p >= t).astype(int)
        bal = balanced_acc(y, yhat)
        acc = float((yhat == y).mean())
        if bal > best_bal:
            best_bal = bal
            best_t = float(t)
            best_acc = acc

    return best_t, float(best_bal), float(best_acc)
Then inside run_transfer_with_platt() after you compute p_raw, p_platt, y_test, etc., also compute CAL probabilities:

Right after p_raw = sigmoid(z_test) and p_platt = ..., compute:

# CAL probabilities (for threshold selection)
z_cal_for_p = z_cal  # already flipped
p_cal_raw = sigmoid(z_cal_for_p)
p_cal_platt = sigmoid(a * z_cal0 + b)

# Choose thresholds on CAL
t_raw, bal_cal_raw, acc_cal_raw = best_threshold_balanced_acc(y_cal, p_cal_raw)
t_pl,  bal_cal_pl,  acc_cal_pl  = best_threshold_balanced_acc(y_cal, p_cal_platt)

# Apply thresholds to TEST
yhat_raw_t = (p_raw >= t_raw).astype(int)
yhat_pl_t  = (p_platt >= t_pl).astype(int)

bal_raw_t = balanced_acc(y_test, yhat_raw_t)
bal_pl_t  = balanced_acc(y_test, yhat_pl_t)

acc_raw_t = float((yhat_raw_t == y_test).mean())
acc_pl_t  = float((yhat_pl_t == y_test).mean())
And add these fields to out_rows.append({...}):

"tstar_raw": t_raw,
"tstar_platt": t_pl,
"bal_raw_tstar": bal_raw_t,
"bal_platt_tstar": bal_pl_t,
"acc_raw_tstar": acc_raw_t,
"acc_platt_tstar": acc_pl_t,
"bal_cal_raw_tstar": bal_cal_raw,
"bal_cal_platt_tstar": bal_cal_pl,
That’s it.