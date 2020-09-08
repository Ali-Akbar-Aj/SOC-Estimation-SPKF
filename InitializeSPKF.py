import numpy as np
from SOCfromOCVtemp import SOCfromOCVtemp

def initSPKF(v0, T0, SigmaX0, SigmaV, SigmaW, model):
    "State Initialization"
    spkfData = {}
    ir0= 0
    hk0 = 0
    SOC0 = SOCfromOCVtemp(v0, T0, model)
    spkfData['irInd'] = 0
    spkfData['hkInd'] = 1
    spkfData['zkInd'] = 2
    spkfData['xhat'] = {}
    spkfData['xhat'][0] = ir0
    spkfData['xhat'][1] = hk0
    spkfData['xhat'][2] = SOC0

    "Covariance matrix"
    spkfData['SigmaX'] = SigmaX0
    spkfData['SigmaV'] = SigmaV
    spkfData['SigmaW'] = SigmaW
    noise_array = np.array([SigmaW,SigmaV])
    spkfData['Snoise'] = np.real(np.linalg.cholesky((np.diagflat(noise_array))))
    spkfData['Qbump'] = 5

    Nx = len(spkfData['xhat']) ##Length of state vector
    Ny = 1 ##Length of measurement vector
    Nu =1 ##Length of input vector
    Nw = len(SigmaW) ## Process noise vector length
    Nv = len(SigmaV) ## Sensor noise vector length
    Na = Nx + Nw + Nv ## Augmented state vector length
    #print(Na)
    spkfData['Nx'] = Nx
    spkfData['Ny'] = Ny
    spkfData['Nu'] = Nu
    spkfData['Nw'] = Nw
    spkfData['Nv'] = Nv
    spkfData['Na'] = Na

    h = np.sqrt(3) ## Tuning factor
    spkfData['h'] = h
    Weight1 = ((h*h)-Na)/(h*h)
    Weight2 = 1/(2*h*h)
    s = Weight2*np.ones((2*Na,1))
    spkfData['Wm'] = np.insert(s,0,Weight1) ##Mean
    spkfData['Wc'] = spkfData['Wm'] ## Covariance

    spkfData['priorI'] = 0 ## Previous value of current
    spkfData['signIk'] = 0 ## Previous Sign of current

    spkfData['model'] = model

    return spkfData




