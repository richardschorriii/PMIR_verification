#!/usr/bin/env python3
"""
PASS47 - PMIR Verification Test
Extracted from ChatGPT transcript (line 94973)
Length: 6659 characters
"""

PS C:\Users\veilbreaker\string_theory> python -c "import pandas as pd; df=pd.read_csv(r'$O47A\pass45_seedlevel_regression_summary.csv'); print(df.to_string(index=False))"
                    model  n_rows       r2   beta_c1  beta_med_c1  beta_lo_c1  beta_hi_c1  beta_logN  beta_med_logN  beta_lo_logN  beta_hi_logN  beta_probe_eps  beta_med_probe_eps  beta_lo_probe_eps  beta_hi_probe_eps  beta_topo_is_grid  beta_med_topo_is_grid  beta_lo_topo_is_grid  beta_hi_topo_is_grid  beta_gap_cv  beta_med_gap_cv  beta_lo_gap_cv  beta_hi_gap_cv
       M0_logY~1+controls    1800 0.345512 -1.822866    -1.812023   -3.208422   -0.376211  -0.766311      -0.767460     -0.950189     -0.587342       20.182597           20.176041          18.685626          21.729260           1.659035               1.660960              1.455362              1.860516          NaN              NaN             NaN             NaN
M1_logY~1+controls+gap_cv    1800 0.345610 -1.972790    -1.971135   -3.427705   -0.512653  -0.775474      -0.777154     -0.967761     -0.591047       20.182597           20.172533          18.690743          21.726862           1.021674               0.977877             -1.017758              3.061409     0.346241         0.365419       -0.764168        1.452520
         M2_logY~1+gap_cv    1800 0.094948 -6.944160    -6.945346   -7.145548   -6.732043        NaN            NaN           NaN           NaN             NaN                 NaN                NaN                NaN                NaN                    NaN                   NaN                   NaN     0.889614         0.889683        0.763609        1.014592
PS C:\Users\veilbreaker\string_theory> python -c "import pandas as pd; g=pd.read_csv(r'$O47A\pass45_seedlevel_group_corr.csv'); print('group rows',len(g)); print(g.head(30).to_string(index=False))"
group rows 60
probe_dir_mode probe_mode  probe_eps  n_rows      corr   x_mean     y_mean
       fiedler  add_to_dx      0.002      30  0.984139 1.555198  -8.948468
       fiedler  add_to_dx      0.005      30  0.973987 1.555198  -8.088956
       fiedler  add_to_dx      0.010      30  0.954453 1.555198  -7.431932
       fiedler  add_to_dx      0.020      30  0.917177 1.555198  -6.575300
       fiedler  add_to_dx      0.035      30  0.930196 1.555198  -6.025142
       fiedler  add_to_dx      0.050      30  0.919590 1.555198  -5.737853
       fiedler  add_to_dx      0.070      30  0.917033 1.555198  -5.433514
       fiedler  add_to_dx      0.100      30  0.933561 1.555198  -5.067411
       fiedler  add_to_dx      0.140      30  0.915097 1.555198  -4.764336
       fiedler  add_to_dx      0.200      30  0.913739 1.555198  -4.406459
       fiedler   add_to_x      0.002      30  0.925483 1.555198  -5.923083
       fiedler   add_to_x      0.005      30  0.933561 1.555198  -5.067411
       fiedler   add_to_x      0.010      30  0.913739 1.555198  -4.406459
       fiedler   add_to_x      0.020      30  0.796752 1.555198  -3.908344
       fiedler   add_to_x      0.035      30  0.883931 1.555198  -3.146145
       fiedler   add_to_x      0.050      30  0.923861 1.555198  -2.647651
       fiedler   add_to_x      0.070      30  0.924329 1.555198  -2.317155
       fiedler   add_to_x      0.100      30  0.921996 1.555198  -2.037128
       fiedler   add_to_x      0.140      30  0.914613 1.555198  -1.787395
       fiedler   add_to_x      0.200      30  0.906276 1.555198  -1.531428
        random  add_to_dx      0.002      30 -0.000863 1.555198 -11.314778
        random  add_to_dx      0.005      30 -0.043103 1.555198 -10.424359
        random  add_to_dx      0.010      30 -0.115753 1.555198  -9.823276
        random  add_to_dx      0.020      30 -0.149529 1.555198  -9.153909
        random  add_to_dx      0.035      30 -0.041174 1.555198  -8.420589
        random  add_to_dx      0.050      30 -0.005549 1.555198  -8.026113
        random  add_to_dx      0.070      30 -0.010491 1.555198  -7.685240
        random  add_to_dx      0.100      30 -0.129237 1.555198  -7.445676
        random  add_to_dx      0.140      30 -0.110166 1.555198  -7.100896
        random  add_to_dx      0.200      30  0.279024 1.555198  -6.333913
PS C:\Users\veilbreaker\string_theory> $O47B = "analysis_transfer\PMIR_TIME_LOCK_MASTER\runs_eps_sweep\probe_invariance\pass47_seedlevel_topology_controls\B_gapcvXgrid"
PS C:\Users\veilbreaker\string_theory> New-Item -ItemType Directory -Force -Path $O47B | Out-Null
PS C:\Users\veilbreaker\string_theory>
PS C:\Users\veilbreaker\string_theory> python .\pass45_seedlevel_regression.py 
>>   --seed_join_csv "$O47\pass47_seedlevel_join_aug.csv" 
>>   --outdir $O47B 
>>   --predictor gapcv_x_grid 
>>   --controls logN probe_eps topo_is_grid gap_cv 
>>   --boot_reps 5000 
>>   --seed 1337
[PASS45] complete
[PASS45] wrote analysis_transfer\PMIR_TIME_LOCK_MASTER\runs_eps_sweep\probe_invariance\pass47_seedlevel_topology_controls\B_gapcvXgrid\pass45_seedlevel_regression_summary.csv
[PASS45] wrote analysis_transfer\PMIR_TIME_LOCK_MASTER\runs_eps_sweep\probe_invariance\pass47_seedlevel_topology_controls\B_gapcvXgrid\pass45_seedlevel_group_corr.csv
[PASS45] wrote analysis_transfer\PMIR_TIME_LOCK_MASTER\runs_eps_sweep\probe_invariance\pass47_seedlevel_topology_controls\B_gapcvXgrid\pass45_seedlevel_regression_summary.txt
PS C:\Users\veilbreaker\string_theory> Get-Content "$O47B\pass45_seedlevel_regression_summary.txt"
PASS45 â€” seed-level spectral regression

seed_join_csv: analysis_transfer\PMIR_TIME_LOCK_MASTER\runs_eps_sweep\probe_invariance\pass47_seedlevel_topology_controls\pass47_seedlevel_join_aug.csv
predictor: gapcv_x_grid
controls: ['logN', 'probe_eps', 'topo_is_grid', 'gap_cv']
boot_reps: 5000 seed: 1337
rows used (after dropna): 1800

M0_logY~1+controls
  r2: 0.345610
  c1: beta=-1.97279  med=-1.97113  CI=[-3.4277,-0.512653]
  logN: beta=-0.775474  med=-0.777154  CI=[-0.967761,-0.591047]
  probe_eps: beta=20.1826  med=20.1725  CI=[18.6907,21.7269]
  topo_is_grid: beta=1.02167  med=0.977877  CI=[-1.01776,3.06141]
  gap_cv: beta=0.346241  med=0.365419  CI=[-0.764168,1.45252]

M1_logY~1+controls+gapcv_x_grid
  r2: 0.355096
  c1: beta=-4.51107  med=-4.51813  CI=[-6.14485,-2.90583]
  logN: beta=-0.404882  med=-0.404215  CI=[-0.615514,-0.193201]
  probe_eps: beta=20.1826  med=20.1759  CI=[18.7026,21.7214]
  topo_is_grid: beta=679  med=678.847  CI=[438.272,941.109]
  gap_cv: beta=-0.106412  med=-0.070732  CI=[-1.2265,1.0031]
  gapcv_x_grid: beta=-273.528  med=-273.336  CI=[-379.375,-176.41]

M2_logY~1+gapcv_x_grid
  r2: 0.096650
  c1: beta=-6.38969  med=-6.38983  CI=[-6.53275,-6.24029]
  gapcv_x_grid: beta=0.669781  med=0.669989  CI=[0.574636,0.763045]