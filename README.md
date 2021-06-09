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


### Branch - IJUQ

This repository is used to generate all the tables and figures in the following paper:

$L^p$ Convergence of Approximate Maps and Probability Densities for Forward and Inverse Problems in Uncertainty Quantification.

### Branch - dissertation

This branch is used to generate all the tables and figures in Chapter 3 of my dissertation.



