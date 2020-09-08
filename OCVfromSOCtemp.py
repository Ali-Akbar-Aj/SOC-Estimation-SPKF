import numpy as np

"This part of code helps in estimating open circuit voltage (OCV)"
"from the available data of state of charge (SOC) and temperature (temp) "

def OCVfromSOCtemp(soc,temp,model):
    soccol = soc.flatten() ##Convert SOC into a column vector
    SOC = model['SOC'].flatten()
    OCV0 = model['OCV0'].flatten()
    OCVrel = model['OCVrel'].flatten()

    if np.isscalar(temp):    ## If temp variable is scalar then replicatce the same temp for all SOC values
        tempcol = temp*np.ones(soccol.shape)
    else: ## Convert temp into column vector
        tempcol = temp.flatten()
        if tempcol.shape != soccol.shape:
            print('Function temp and SOC must have same imputs or temp must be scalar')
    diffSOC = SOC[1] - SOC[0]
    ocv = np.zeros(soccol.shape) ## reserve space for OCV
    I1 = np.array(np.where(soccol <= SOC[0])) ## Indices of SOC's smaller than the lowest value of SOC available from data
    I2 = np.array(np.where(soccol >= SOC[-1])) ## Indices of SOC's higher than the largest value of SOC available from data
    I3 = np.array(np.where((soccol > SOC[0]) & (soccol < SOC[-1]))) ## Indices of all SOC's in between
    I6 = np.array(np.where(np.isnan(soccol))) ##Indices for entries which are not a number

    ## For SOC's which are lower than the lowest SOC data point available extrapolate at the lower end
    if I1.size != 0:
        dv = (OCV0[1] + np.multiply(tempcol,OCVrel[1])) - (OCV0[0] + np.multiply(tempcol,OCVrel[0]))
        ocv[I1] = (np.multiply((soccol[I1] - SOC[0]),dv[I1]))/diffSOC + OCV0[0] + np.multiply(tempcol[I1],OCVrel[0])

    ## For SOC's which are higher than the largest SOC data point available extrapolate at the upper end
    if I2.size != 0:
        dv = (OCV0[-1] + np.multiply(tempcol,OCVrel[-1])) - (OCV0[-2] + np.multiply(tempcol,OCVrel[-2]))
        ocv[I2] = (np.multiply((soccol[I2] - SOC[-1]),dv[I2]))/diffSOC + OCV0[-1] + np.multiply(tempcol[I2],OCVrel[-1])

    ## Linear interpolatin for all the values in between
    I4 = (soccol[I3]-SOC[0])/diffSOC
    I5 = np.floor(I4)
    I5 = I5.astype(int)
    I45 = I4 - I5
    omI45 = 1 - I45
    ocv[I3] = np.multiply(OCV0[I5],omI45) + np.multiply(OCV0[I5+1],I45)
    ocv[I3] = ocv[I3] + np.multiply(np.multiply(tempcol[I3],OCVrel[I5]),omI45) +np.multiply(OCVrel[I5+1],I45)
    ocv[I6] = 0
    ocv = np.reshape(ocv,soc.shape)
    return ocv


