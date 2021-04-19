# Unsupervised sample selection for multivariate calibration with PLS models
This repository contains the implementation of different algorithms of unsupervised sample selection for multivariate calibration models in chemometrics. The methods that were reported and tested were hierarchical clustering, duplex, kennard stone, puchwein and d-optimal design of experiments. Other methods are available also in the current repository.

The use of d-optimal designs is fetched from R with the [AlgDesign package](https://cran.r-project.org/web/packages/AlgDesign/index.html). This package needs to be installed manually to use the current d-optimal implementation. 


|    Package  |   Version  | Priority   |  Depends |  
|------------ | -----------|------------|----------|
|"AlgDesign"  |  "1.2.0"   |      NA    |      NA  |


The methods can be found in the **methods** folder. There is a class for sample selection with the mentioned methods and there is a module with the simpls algorithm for PLSR model building using numba for large data sets and a high number of latent variables.

A protocol for unsupervised sample selection and evaluation can be found in the **scripts** folder. The other scripts and folders correspond to the reported results in the original publication. 

For reproducibility of the original publication, please use the environment.yml file. 

