# RTs-for-AdaptiveData

## Code layout
Our code is comprised of driver runscripts, simulation-specific scripts (containing functions particular to the environment, adaptive assignment algorithm, and resampling procedure being considered in the particular simulation), and global scripts used across all simulations (i.e., test statistics as well as the weighted MC and unweighted MCMC randomization tests themselves). We describe these three types of scripts below. 

### Driver runscripts
There are three driver runscripts to be used to reproduce the simulations run in Simulation 5: 1. `testing_driver.py`, 2. `interval_driver.py`, and 3. `conformal_interval_driver_share.py`. These three scripts are, respectively, responsible for running 1. hypothesis testing simulations, 2. confidence and conformal prediction interval construction (without sample sharing) simulations, and 3. conformal prediction interval construction with sample sharing simulations. 
