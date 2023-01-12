# RTs-for-AdaptiveData

## Code layout
Our code is comprised of driver runscripts, simulation-specific scripts (containing functions particular to the environment, adaptive assignment algorithm, and resampling procedure being considered in the particular simulation), and global scripts used across all simulations (i.e., test statistics as well as the weighted MC and unweighted MCMC randomization tests themselves). We describe these three types of scripts below. 

### Driver runscripts
There are three driver runscripts to be used to reproduce the simulations run in Simulation 5: 1. `testing_driver.py`, 2. `interval_driver.py`, and 3. `conformal_interval_driver_share.py`. These three scripts are, respectively, responsible for running 1. hypothesis testing simulations, 2. confidence and conformal prediction interval construction (without sample sharing) simulations, and 3. conformal prediction interval construction with sample sharing simulations. 

All driver runscripts take as command line input a "style" (i.e., the type of resampling algorithm), which is typically abbreviated as the first letter in the resampling procedure's name. The table below delineates the correspondence between style and name of the corresponding resampling procedure:

| Style  | Resampling procedure name |
| ------------- | ------------- |
| u  | Both $\text{uniform}_{\pi}$ and $\text{uniform}_X$  |
| 1  | 5.1.2  |
| 2  | 5.1.1. |
| 3  | 5.2.3. |
| 4  | 5.2.2. |

Furthermore, the testing driver runscript takes as one of its command line arguments the "setting number" which corresponds to the simulation/environment being considered. The table below gives the correspondence between setting number and the corresponding simulation subsection of the paper:

| Setting number  | Simulation subsection |
| ------------- | ------------- |
| 0  | 5.2.1  |
| 1  | 5.1.2  |
| 2  | 5.1.1. |
| 3  | 5.2.3. |
| 4  | 5.2.2. |

The full command line input for `testing_driver.py` comprises 10 arguments: 1. the number of desired trials/repetitions to be run, 2. horizon $T$, 3. job/seed number, 4. adaptive assignment algorithm name, 5. if the null hypothesis is being used
