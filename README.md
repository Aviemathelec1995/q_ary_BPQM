Code to simulate Density Evolution for LDPC and polar codes based on the paper- [Belief Propagation with Quantum Messages for Symmetric Q-ary Pure-State Channels](https://arxiv.org/pdf/2601.21330)




Example command for LDPC

`python qudit_psc_ldpc_threshold.py --q 5 --dv 3 --dc 6 --ns 3000 --depth 50 --tol 0.05`

Outputs threshold lambda using BPQM for (dv,dc) regular LDPC code on symmetric q-ary PSC with Gram matrix eigenlist of the form [lambda,q-lambda/(q-1),....q-lambda/(q-1)]

Example command for Polar

`python qudit_psc_polar_plots.py design-curve --q 3 --n_list 7,8,9,10 --lam0 2.4,0.45,0.15 --Npop 2000 --out design_curve.png`


`python qudit_psc_polar_plots.py rate-vs-capacity --q 3 --n_list 8,10,12 --d_budget 0.1 --Npop 4000 --out rate_vs_cap.png`

Outputs design curve and rate vs capacity plots for polar codes on symmetric q-ary PSC with given eignelist

