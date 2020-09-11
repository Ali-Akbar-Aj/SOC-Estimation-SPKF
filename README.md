# SOC-Estimation-SPKF
This project helps in estimating SOC of cell using Sigma Point Kalman Filter (SPKF) and Enhanced Self Correcting (ESC) cell model

# Cell Model
Enhanced Self Correcting (ESC) cell model is used to replicate the cell behaviour and the file "E2model" has all the values of the ESC model circuit

# Loading data
The cell charge/discharge dynamic(DYN) data at different temperature is uploaded in the file named "E2_DYN_XXX", where XXX represent different temperatures

# Loadmat
This part of the code helps in retrieving the stored cell data in the MAT folder to python dictionaries

# MasterSPKF
The code act as a central spine in running the Sigma Point Kalman Filter (SPKF) algorithm for the provided data

# InitializeSPFK
Helps in initializing the variables and the function is called by the MasterSPKF code

# IterationSPKF
As the name suggest, this part of the code is called by MasterSPKF at every iteration to update the SOC estimate

# OCVfromSOCtemp and SOCfromOCVtemp
This helper code gets the OCV (SOC) data from the available SOC (OCV) and temperature data
