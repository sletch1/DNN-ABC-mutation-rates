# GPS-ABC estimator for piece-wise mutation rate with exact simulators

This folder contains the GPS-ABC estimator under the piece-wise constant mutation rate assumption. The embedded simulator performs "exact simulation" based on a two-type Markov branching process (MBP). The detailed discussions of the two simulators could be found in the manuscript.
* countsizeBDtree2: the exact simulator based on the piece-wise constant mutation rate assumption.
* ABC_mu2: The GPS-ABC estimator.
* simulation2_main: The main script to generate data from the simulator `countsizeBDtree2` and apply GPS-ABC estimator on the simulated data.

*Note*: The detailed model specifications are described as comments in the script.  
