# Lp

This **master** repository is used to generate all the tables and figures in the following paper:

[*Convergence of Probability Densities using Approximate Models for Forward and Inverse Problems in Uncertainty Quantification: Extensions to $L^p$*](https://arxiv.org/abs/2001.04369)


***PS: There are 8 tables and 5 Figures in total.***

----
**To generate results in the paper, you can follow the following guidance.**


## Dependencies
All the code is current run on Python 3.6 mainly using NumPy, SciPy, matplotlib, OS, DOLFIN and math.


## Branches on this repository

### Branch - master

- "ODE example.py"

This file is used to generate Table 1-3, Fig 1-2. After running this file using command:
    
    python ODE\ example.py

it will automatically print out data in Table 1-3 on the window. Besides that, a directory named "images" will be created automatically (if it does not exist) including all the figures you just generated.


- "PDE example.py"

This file is used to generate Table 4-5, Fig 3-4. You will need data from "Data" directory to help get results. (The way to generate these data will be explained later.) After running this file using command:
    
    python PDE\ example.py

it will automatically print out data in Table 4-5 on the window. Besides that, a directory named "images" will be created automatically (if it does not exist) including all the figures you just generated.


- "Appendix A.py"

This file is used to generate Table A.6-8, Fig A.5. After running this file using command:
    
    python Appendix\ A.py

it will automatically print out data in Table 6-8 on the window. Besides that, a directory named "images" will be created automatically (if it does not exist) including all the figures you just generated.


- "GenerateData.py"

This file is used to generate data required to run "PDE example.py". Note: you can run this file using command:

    python GenerateData.py
    
This will automatically create directory "Data" (if it does not exist) including all the needed data. You can skip this command and directly use the existing directory "Data" when you run "PDE example.py" to save your time.


### Branch - supplemental

There are two directories in this branch, "Data" and "images".

- "Data" directory

It includes all the data needed to run "PDE example.py" in the master branch.

To manually generate these data, you can run "GenerateDate.py" in the master branch.

- "images" directory

It includes all the figures in the paper.

To manually generate these figures, you can run "ODE example.py" to get Fig1-2, "PDE example.py" to get fIG 3-4 and "Appendix A.py" to get FigA.5.

