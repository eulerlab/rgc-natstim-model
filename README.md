# rgc-natstim-model
## Description
Companion repository to publication "[A chromatic feature detector in the retina signals visual context changes](https://www.biorxiv.org/content/10.1101/2022.11.30.518492v2)".

We trained CNN models (*digital twins*) of mouse retinal processing of naturalistic stimuli. We then used the models to analyse neuronal stimulus selectivities *in-silico* and found a selectivity for chromatic contrast in a type of contrast-suppressed retinal ganglion cell (RGC). Based on this feature, we proposed a role in detecting visual context changes for this RGC type.

This repository contains the code to reproduce the analyses and figures presented in the paper. 

## How to use this repository
1. Clone this repository, navigate to its directory, and install it via  
`pip install .`
2. Download the data and model files from [G-Node ](https://gin.g-node.org/lhoefling/rgc-natstim). Update the `base_directory` in `rgc_natstim_model/constants/paths.py` to point to the respective directory on your machine. 
### Reproducing figures 
 
### Training models

