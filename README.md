# LIFD
_By Sofia Sosa Fiscella_

This repository introduces PINT implementations of the Inverse-Frequency Depedent (IFD) and Legendre-Inverse Frequency Dependent (LIFD) models of pulse profile evolution with frequency. The corresponding Python classes can be found in `IFD_class.py` and `LIFD_class.py`. Moreover, in `main.py` we present a minimal script showing how to attach these classes to a timing model, perform a timing fit to NG15 observations, and analyze the resulting timing models. In order to do so: 

1. Download the NANOGrav 15-year dataset from https://zenodo.org/records/16051178 and place the resulting folder to the same directory as the Python scripts. 
2. For each pulsar, we re-calculated the nominal DM value by using the Golden Section Search algorithm to find the DM value that gives the maximum average S/N for the pulsar's `.ff` files (see `DM_search.py` in the folder `DM_calculations`.) That value is then read by the main script, `main.py`. 
3. In `main.py`, select:

   1. The name of the pulsar you want to analyze.
   2. If you want to use real or simulated data.
   3. If you want to generate intermediate residual plots.

3. Run `python main.py`. As a result, you will obtain `.npy` files containing the FD/IFD/LIFD coefficient values, and some intermediate plots..
