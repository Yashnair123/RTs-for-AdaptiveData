# RTs-for-AdaptiveData

## Code layout
Our code is comprised of 1. key runscripts that we run across many simulations which we call "driver" runscripts; 2. simulation-specific scripts (containing functions particular to the environment, adaptive assignment algorithm, and resampling procedure being considered in the particular simulation); and 3. global scripts used across all simulations (i.e., test statistics as well as the weighted MC and unweighted MCMC randomization tests themselves). We describe these three types of scripts below. 

### Driver runscripts
There are three driver runscripts to be used to reproduce the simulations run in Section 5: 1. `testing_driver.py`, 2. `interval_driver.py`, and 3. `conformal_interval_driver_share.py`. These three scripts are, respectively, responsible for running 1. hypothesis testing simulations, 2. confidence and conformal prediction interval construction (without sample sharing) simulations, and 3. conformal prediction interval construction with sample sharing simulations. 

All driver runscripts take as command line input a "style" (i.e., the type of resampling algorithm), which is typically abbreviated as the first letter in the resampling procedure's name. The table below delineates the correspondence between style and name of the corresponding resampling procedure:

| Style  | Resampling procedure name |
| ------------- | ------------- |
| `u`  | Both $\text{uniform}_{\pi}$ and $\text{uniform}_X$  |
| `i_X`  | $\text{imitation}_{X}$  |
| `i`  | $\text{imitation}_{\pi}$ |
| `r`  | $\text{re-imitation}_{\pi}$ |
| `c`  | $\text{cond-imitation}_{\pi}$ |
| `uu_X` | $\text{uniform}_{\pi}\text{+}\text{uniform}_X$  |
| `ui_X` | $\text{uniform}_{\pi}\text{+}\text{imitation}_X$  |
| `rui_X` | $\text{restricted-uniform}_{\pi}\text{+}\text{imitation}_X$  |
| `comb` | $\text{combined}_{\pi,X}$  |

The full command line input for `interval_driver.py` comprises 9 arguments: 1. number of trials/repetitions to be run, 2. horizon $T$, 3. job/seed number, 4. adaptive assignment algorithm name, 5. style, 6. whether or not to use the weighted MC randomization test (i.e., `True` if using the weighted MC randomization test and `False` if using the unweighted MCMC randomization test), 7. value of $\epsilon$ to be used (when the adaptive assignment involves an $\epsilon$-greedy action-selection), 8. number of resamples $m$, 9. type of interval (i.e., either `conformal` or `confidence`). Similarly, the command line input for `interval_driver.py` involves 7 arguments: 1. number of trials, 2. horizon $T$, 3. job/seed number, 4. adaptive assignment algorithm name, 5. style, 6. $\epsilon$, 7. number of resamples $m$. 

Two example command line inputs for these to driver runscripts are thus `1000 100 0 epsilon_greedy i True 0.1 100 conformal` and `python interval_driver.py 1000 100 0 epsilon_greedy i 0.1 100`, respectively.

Beyond the inputs described above, the testing driver runscript also takes as one of its command line arguments the "setting number" which corresponds to the simulation/environment in which the test is being run. The table below gives the correspondence between setting number and the corresponding simulation subsection of the paper:

| Setting number  | Simulation subsection in https://arxiv.org/abs/2301.05365|
| ------------- | ------------- |
| `0`  | 5.2.1  |
| `1`  | 5.1.2  |
| `2`  | 5.1.1 |
| `3`  | 5.2.3 |
| `4`  | 5.2.2 |

We additionally list the adaptive assignment algorithm names considered in each of the various settings (`elinucb` stands for $\epsilon$-LinUCB; when $\epsilon = 0$ this is simply the LinUCB adaptive assignment algorithm considered in Section 5 of the paper).

| Setting number  | Adaptive assignment algorithm|
| ------------- | ------------- |
| `0`  | `epsilon_greedy` and `ucb`  |
| `1`  |  `epsilon_greedy` and `ucb` |
| `2`  | `elinucb` and `epsilon_greedy` |
| `3`  | `epsilon_greedy` |
| `4`  | `elinucb` and `epsilon_greedy` |

The full command line input for `testing_driver.py` then comprises 10 arguments: 1. number of trials, 2. horizon $T$, 3. job/seed number, 4. adaptive assignment algorithm name, 5. whether or not the data is generated under the null (i.e., `True` if the data is drawn according to the null and `False` if it is drawn according to the alternative), 6. style, 7. whether or not to use the weighted MC randomization test, 8. $\epsilon$, 9. number of resamples $m$, 10. setting. Thus one example input for this runscript is `1000 100 0 epsilon_greedy False c True 0.1 100 1`.


### Simulation-specific scripts
These scripts are contained in the directories `bandit_non_stationarity`, `factored_bandit_distributional`, `contextual_bandit_distributional`, `mdp_nonstationarity`, and `contextual_non_stationarity` which correspond, respectively, to settings 0, 1, 2, 3, and 4. Each of the scripts in these directories contains code for the adaptive assignment algorithm (as well as the corresponding action-selection probability calculations) in the setting being considered as well as code for the resampling procedures we consider in these settings. So, for example, the file `epsilon_greedy.py` in `bandit_non_stationarity` contains code for the $\epsilon$-greedy adaptive assignment algorithm (and its action-selection probabilities) described in Section 5.2.1 as well as code for resampling from and computing (conditional) resampling probabilities under:
```math
\text{uniform}_{\pi}, \text{ } \text{imitation}_{\pi}, \text{ } \text{re-imitation}_{\pi}, \text{ or } \text{cond-imitation}_{\pi}.
```

### Global scripts
There are two global scripts: `test_statistics.py` and `randomization_tests.py`. The former simply gives code for each of the test statistics in the various simulations/environments considered in the paper. The latter supplies code for the general randomization testing framework, taking as input the test statistic, adaptive assignment algorithm, and resampling procedure from other files.

All simulations in our paper were run in parallel on Harvard's FASRC cluster.
