# Lp

This repository is used to generate all the tables and figures in the following paper:

*$L^p$ Convergence of Approximate Maps and Probability Densities for Forward and Inverse Problems in Uncertainty Quantification*.


***PS: There are 8 tables and 5 Figures in total.***

----

## Dependencies
All the code is current run on Python 3.6 mainly using NumPy, SciPy, matplotlib, OS, DOLFIN and math.


### Current Branch - IJUQ

Both the notebook and py version for all examples are included. They return the same output. But there are more detailed explanations in the notebook version.

- "Almost example.py" and "Almost example.ipynb" 

They are both used to generate Table 1-3, Fig 1. After running this file using command:
    
        python Almost\ example.py

it will automatically print out data in Table 1-3 on the window. Besides that, a directory named "images" will be created automatically (if it does not exist) including all the figures we just generated.


- "ODE example.py" and "ODE example.ipynb"

They are both used to generate Table 4-6, Fig 2-3. After running this file using command:
    
        python ODE\ example.py

it will automatically print out data in Table 4-6 on the window. Besides that, a directory named "images" will be created automatically (if it does not exist) including all the figures you just generated.


- "PDE example.py" and "PDE example.ipynb"

They are both used to generate Table 7-8, Fig 4-5. You will need data from "Data" directory to help get results. (The way to generate these data will be explained below.) After running this file using command:
    
        python PDE\ example.py

it will automatically print out data in Table 7-8 on the window. Besides that, a directory named "images" will be created automatically (if it does not exist) including all the figures you just generated.


- "GenerateData.py", "GenerateData.ipynb" and "GenerateData_ParallelVersion.ipynb"

These are all used to generate data required to run "PDE example.py" or "PDE example.ipynb". Note: you can run this file using command:

        
        python GenerateData.py
    
This will automatically create directory "Data" (if it does not exist) including all the needed data. You can skip this command and directly use the existing directory "Data" when you run "PDE example.py" to save your time.
