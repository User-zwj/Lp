# Lp

This repository is used to generate all the tables and figures in the following paper:

*$L^p$ Convergence of Approximate Maps and Probability Densities for Forward and Inverse Problems in Uncertainty Quantification*.


***PS: There are 8 tables and 5 Figures in total.***

----
**Note: This branch is used to generate the same results in the dissertation but with different numbering and plot formatting**


## Dependencies
All the code is current run on Python 3.6 mainly using NumPy, SciPy, matplotlib, OS, DOLFIN and math.


### Current Branch - dissertation


- "Almost example.py"

This file is used to generate Table 3.1-3.3, Fig 3.1. After running this file using command:
    
    python Appendix\ A.py

it will automatically print out data in Table 3.1-3.3 on the window. Besides that, a directory named "images" will be created automatically (if it does not exist) including all the figures you just generated.


- "ODE example.py"

This file is used to generate Table 3.4-3.6, Fig 3.2-3.3. After running this file using command:
    
    python ODE\ example.py

it will automatically print out data in Table 3.4-3.6 on the window. Besides that, a directory named "images" will be created automatically (if it does not exist) including all the figures you just generated.


- "PDE example.py"

This file is used to generate Table 3.7-3.8, Fig 3.4-3.5. You will need data from "Data" directory to help get results. (The way to generate these data will be explained later.) After running this file using command:
    
    python PDE\ example.py

it will automatically print out data in Table 3.7-3.8 on the window. Besides that, a directory named "images" will be created automatically (if it does not exist) including all the figures you just generated.


- "GenerateData.py"

This file is used to generate data required to run "PDE example.py". Note: you can run this file using command:

    python GenerateData.py
    
This will automatically create directory "Data" (if it does not exist) including all the needed data. You can skip this command and directly use the existing directory "Data" when you run "PDE example.py" to save your time.
