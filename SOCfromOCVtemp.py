import numpy as np

"This part of code helps in estimating state of charge (SOC) "
"from the available data of open circuit voltage (OCV) and temperature (temp) "

def SOCfromOCVtemp (ocv,temp,model):
    ocvcol = ocv.flatten() ##Convert OCV into a column vector
    OCV = model['OCV'].flatten()
    SOC0 = model['SOC0'].flatten()
    SOCrel = model['SOCrel'].flatten()

    if np.isscalar(temp): ## If temp variable is scalar then replicatce the same temp for all SOC values
        tempcol = temp*np.ones(ocvcol.shape)
    else:   ## Convert temp into column vector
        tempcol = temp.flatten()
        if tempcol.shape != ocvcol.shape:
            print('Function temp and OCV must have same imputs or temp must be scalar')

    diffOCV = OCV[1]-OCV[0]
    soc = np.zeros(ocvcol.shape)  ## reserve space for SOC
    I1 = np.array(np.where(ocvcol <= OCV[0])) ## Indices of OCV's smaller than the lowest value of OCV available from data
    I2 = np.array(np.where(ocvcol >= OCV[-1])) ## Indices of OCV's higher than the largest value of OCV available from data
    I3 = np.array(np.where((ocvcol > OCV[0]) & (ocvcol < OCV[-1])))## Indices of all OCV's in between
    I6 = np.array(np.where(np.isnan(ocvcol)))##Indices for entries which are not a number

    ## For OCV's which are lower than the lowest OCV data point available extrapolate at the lower end
    if I1.size != 0:
        dz = (SOC0[1] + np.multiply(tempcol,SOCrel[1])) - (SOC0[0] + np.multiply(tempcol,SOCrel[0]))
        soc[I1] = (np.multiply((ocvcol[I1] - OCV[0]),dz[I1]))/diffOCV + SOC0[0] + np.multiply(tempcol[I1],SOCrel[0])

    ## For OCV's which are higher than the largest OCV data point available extrapolate at the upper end
    if I2.size != 0:
        dz = (SOC0[-1] + np.multiply(tempcol,SOCrel[-1])) - (SOC0[-2] + np.multiply(tempcol,SOCrel[-2]))
        soc[I2] = (np.multiply((ocvcol[I2] - OCV[-1]),dz[I2]))/diffOCV + SOC0[-1] + np.multiply(tempcol[I2],SOCrel[-1])

    ## Linear interpolatin for all the values in between
    I4 = (ocvcol[I3]-OCV[0])/diffOCV
    I5 = np.round(I4)
    I5 = I5.astype(int)
    I45 = I4 - I5
    omI45 = 1 - I45
    soc[I3] = np.multiply(SOC0[I5],omI45) + np.multiply(SOC0[I5+1],I45)
    soc[I3] = soc[I3] + np.multiply(np.multiply(tempcol[I3],SOCrel[I5]),omI45) +np.multiply(SOCrel[I5+1],I45)
    soc[I6] = 0
    soc = np.reshape(soc,ocv.shape)
    return soc
