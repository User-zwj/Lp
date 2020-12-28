# Lp

This repository is used to generate all the tables and figures in the following paper:

*$L^p$ Convergence of Approximate Maps and Probability Densities for Forward and Inverse Problems in Uncertainty Quantification*.


***PS: There are 8 tables and 5 Figures in total.***

----
**To generate results in the paper, you can follow the following guidance.**

## Dependencies
All the code is current run on Python 3.6 mainly using NumPy, SciPy, matplotlib and OS.

## Files
### ODE example.py
This file is used to generate Table 1-3, Fig 1-2. After running this file using command:
    
    python ODE\ example.py

it will automatically print out data in Table 1-3 on the window. Besides that, a directory named "images" will be created automatically (if it does not exist) including all the figures you just generated.


### PDE example.py
This file is used to generate Table 4-5, Fig 3-4. You will need data from "Data" directory to help get results. (The way to generate these data will be explained later.) After running this file using command:
    
    python PDE\ example.py

it will automatically print out data in Table 4-5 on the window. Besides that, a directory named "images" will be created automatically (if it does not exist) including all the figures you just generated.


### Appendix A.py
This file is used to generate Table 6-8, Fig 5. After running this file using command:
    
    python Appendix\ A.py

it will automatically print out data in Table 6-8 on the window. Besides that, a directory named "images" will be created automatically (if it does not exist) including all the figures you just generated.


***PS: you can also directly get all the figures in the "images" branch in this repository.***
