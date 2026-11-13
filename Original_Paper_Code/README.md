# Code for "Estimating mutation rates in a Markov branching process using approximate Bayesian computation"

This software package includes the source code (mostly in MATLAB) for our manuscript "Estimating mutation rates in a Markov branching process using approximate Bayesian computation". There are three parts in the package: I. ABC-based estimator and MOM/MLE in this main folder, II. Simulation study 1 (for the constant mutation scenario) and Simulation study 2 (for the piece-wise constant mutation scenario) under subfolder `simulation/`, and III. Real data analysis under subfolder `real/`. 

## Estimating a constant mutation rate
* ABC-based estimator with "exact" simulator:
    * **fluc_exp1.m**: The "exact" simulation. Generate fluctuation data in parallel cultures from bMBP model with constant mutation rate. Used everywhere.
    * **ABC_fluc_exp1.m**: Estimate mutation rate by ABC for fluctuation data with constant mutation rate. Used in `simulation/simu1gendata_orisim.m` and `simulation/simu1_orisim_statlnx9.m`.
    * trainGPS.m: Train GPS model for fluctuation data with constant mutation rate. Used in `ABC_fluc_exp1.m`, also in `demoGPS_fluc_exp1_rev.m`.
    * tnrnd.m: Generate random sample from truncated normal. Used in `ABC_fluc_exp1.m`.
    * mut_bMBP.m: Generate (z, x) data in a single culture from bMBP model with constant mutation rate. Used in `fluc_exp1.m`.
    * selsummary_fluc_exp1.m: Compare the response curves of three different summary statistics (Fig. 1).
    * demoGPS_fluc_exp1_rev.m: Demonstrate the performance of GP regression (Fig. 2).
* ABC-based estimator with "fast" simulator
    * **fluc_exp1_rev.m**: The "fast" simulator. Generate fluctuation data in parallel cultures from approximated bMBP model with constant mutation rate. Differential growth between mutants and normal cells is allowed. Used everywhere.
    * **ABC_mu1.m**: GPS-ABC estimator based on constant mutation rate assumption. Used in `simulation/simulation1_server.m` and `real/werngren_server1.m`.
    * **ABC_MCMC.m**: ABC-MCMC estimator based on constant mutation rate assumption. Used in `simulation/simulation1_MCMC_server.m`.
    * **ABC_mu1a.m**: GPS-ABC estimator to estimate constant mutation rate and relative growth rate of mutants vs. normal cells. Used in `real/werngren_server1a.m`.
    * get_IniX1*.m: Generate initial training samples for fitting GP surrogate model. Used in ABC_mu1*.m. 
    * get_theta1.m and get_theta1a.m: Propose from the proposal distribution. Used in `ABC_mu1.m` and `ABC_mu1a.m`, respectively.
    * tnorm.m: Generate random sample from truncated normal. Used in `ABC_mu1.m`, `ABC_mu1a.m`, `ABC_mu2.m` and `ABC_mu2a.m`.
    * unconditionError: Calculate unconditional error of making an accept/reject decision. Used in `ABC_mu1.m`, `ABC_mu1a.m`, `ABC_mu2.m` and `ABC_mu2a.m`.
    * mut_bMBP_rev.m: Generate approximated (z, x) data in a single culture based on the conditional distribution of the occurring time of mutations.
* MOM/MLE estimator:
    * **MOMMLE_fluc_exp1.m**: Estimate mutation rate by MOM/MLE for fluctuation data with constant mutation rate. Used in` `simulation/simulation1` and `simulation/simu1_orisim_statlnx9.m`.

## Estimating parameters in a piece-wise constant mutation rate function
* ABC-based estimator with "exact" simulator:

   Functions and detailed instructions could be found in the subfolder `exact simulator/`.
   
* ABC-based estimator with "fast" simulator
    * **fluc_exp2_rev.m**: The "fast" simulator with a piece-wise constant mutation rate. Generate fluctuation data in parallel cultures . Differential growth between mutants and normal cells is allowed. Used everywhere.
    * **ABC_mu2a.m**: GPS-ABC estimator to esimate four parameters in a piece-wise constatnt mutation rate function, p1, p2, transition time &tau; and relative growth rate &delta;. Used in `real/werngren_server2a`.
    * ABC_mu2.m: GPS-ABC estimator to estimate three parameters in a piece-wise constant mutation rate function, p1, p2 and tarnsition time &tau;. 
    * get_IniX2.m and get_Inix2a.m: Generate initial training samples for fitting GP surrogate model. Used in `ABC_mu2.m` and `ABC_mu2a.m`, respectively. 
    * get_theta2.m and get_theta2a.m: Propose from the proposal distribution. Used in `ABC_mu2.m` and `ABC_mu2a.m`, respectively.
    * mut2stage_bMBP_rev.m: Generate approximated (z, x) data in a single culture based on the conditional distribution of the occurring time of mutations.
  
*Note*: The detailed model specifications are described as comments in the script.  
