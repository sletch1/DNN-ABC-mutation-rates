# Application of GPS-ABC on real data
This folder includes the source code for the application section in the manuscript. We analyzed 13 datases from a fluctuation experiment in a study of durg resisance [1]. The GPS-ABC esitmation of the mutation rate is conducte under assumptionsof both constant mutation rate (with/without different growth) and two-stage mutations with differential growth. 

## Source code directory
* werngren_main: manually insert data (as shown in Table 5 in the manuscript), apply GPS-ABC estimators defined in ABC_mu1, ABC_mu1a and ABC_mu2a onto the datasets, and generate simulated data using estimated parameters for model selection.
* werngren_server1: a function to apply `ABC_mu1` on the specified to estimate a constant mutation rate, given transition step width, initial value, expected growth rate and number of posterior samples. This function was used to test if the estimation is sensitive to the choice of initial values and hyperparameters.
* werngren_server1a: Apply `ABC_mu1a` on the specified dataset to estimate mutation rate and relative growth rate.
* werngren_server2a: Apply `ABC_mu2a` on the specified dataset to estimate mutation rates p1 and p2, transition time &tau;, and relative growth rate &delta;.
* model_selection: summarize the numbers shown in Table 7 from the simulated data with estimated parameters.

## Reference
[1] Werngren, J., & Hoffner, S. E. (2003). Drug-susceptible Mycobacterium tuberculosis Beijing genotype does not develop mutation-conferred resistance to rifampin at an elevated rate. Journal of clinical microbiology, 41(4), 1520-1524.
