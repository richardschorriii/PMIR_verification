#!/usr/bin/env python3
"""
PASS35 - PMIR Verification Test
Extracted from ChatGPT transcript (line 83443)
Length: 2613 characters
"""

import pandas as pd
df=pd.read_csv(r'$M35')
df=df[df['pass_id']=='P2_PROBE_INVARIANCE'].copy()
df['censored']=pd.to_numeric(df['censored'],errors='coerce')
G=(df.groupby(['topology','N','probe_eps'],as_index=False)
     .agg(cens_frac=('censored','mean'), rows=('censored','size')))
G.to_csv(r'$CENS',index=False)
print('[OK] wrote', r'$CENS', 'rows', len(G))
print(G.sort_values('cens_frac').head(15).to_string(index=False))
"
Step 2 — choose threshold and filter
Pick a threshold that keeps enough blocks. A solid default is cens_frac <= 0.50 unless it drops too much coverage.

$M35F = "analysis_transfer\PMIR_TIME_LOCK_MASTER\runs_eps_sweep\probe_invariance\pass35_censor_min\tauhalf_master_p35_FILT.csv"
$THR = 0.50

python -c "
import pandas as pd
df=pd.read_csv(r'$M35')
df=df[df['pass_id']=='P2_PROBE_INVARIANCE'].copy()
df['censored']=pd.to_numeric(df['censored'],errors='coerce')
C=pd.read_csv(r'$CENS')
keep=set(map(tuple, C[C.cens_frac<=float($THR)][['topology','N','probe_eps']].to_numpy()))
df['__key']=list(map(tuple, df[['topology','N','probe_eps']].to_numpy()))
df=df[df['__key'].isin(keep)].drop(columns='__key')
df.to_csv(r'$M35F',index=False)
print('[OK] wrote', r'$M35F')
print('rows:',len(df))
print('blocks kept:',len(keep))
print(df.groupby(['topology','N']).size().reset_index(name='rows').to_string(index=False))
"
Step 3 — rerun PASS33 on the filtered master
$O35P33 = "analysis_transfer\PMIR_TIME_LOCK_MASTER\runs_eps_sweep\probe_invariance\pass35_censor_min\pass33_after_filter"

python .\pass33_probe_invariance_auc.py `
  --in_csv   $M35F `
  --outdir   $O35P33 `
  --score_col auc_delta `
  --topo_a rr --topo_b grid2d_periodic `
  --reps 20000 --seed 1337 `
  --agg mean
Step 4 — build PASS34 input from filtered PASS33 summary and run PASS34
$P33SUM_F = "$O35P33\pass33_summary_by_dir_probe_topoN_eps.csv"
$P34IN_F  = "$O35P33\pass34_in_from_pass33_summary.csv"
$O35P34   = "analysis_transfer\PMIR_TIME_LOCK_MASTER\runs_eps_sweep\probe_invariance\pass35_censor_min\pass34_after_filter"

python -c "
import pandas as pd
df=pd.read_csv(r'$P33SUM_F')
df=df.rename(columns={'mean_score':'delta_mean_a_minus_b'})
G=(df.groupby(['topology','N','probe_eps'],as_index=False)
     .agg(delta_mean_a_minus_b=('delta_mean_a_minus_b','mean')))
G.to_csv(r'$P34IN_F',index=False)
print('[OK] PASS34 input rows:',len(G))
print(G.head(12).to_string(index=False))
"

python .\pass34_scaling_regression_auc.py `
  --in_csv  $P34IN_F `
  --outdir  $O35P34 `
  --topo_ref rr `
  --boot_reps 5000 `
  --seed 1337
Answering your earlier “will either run for hours?”
PASS34 (what you just ran): minutes