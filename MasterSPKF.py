import numpy as np
import scipy.io as sio
from SOCfromOCVtemp import SOCfromOCVtemp
from OCVfromSOCtemp import OCVfromSOCtemp
from InitializeSPKF import initSPKF
from IterationSPKF import iterSPKF
from loadmat import loadmat
from RetrieveParamESCmodel import getParamESC
from matplotlib import pyplot as plt

"Load ESC battery model file"
E2model = loadmat('E2model.mat')
model = E2model['model']

"Load cell test data"
E2_DYN_15_P05 = loadmat('E2_DYN_15_P05')
DYNData = E2_DYN_15_P05['DYNData']
T = 5 ##Temperature = 5 Degree

time = DYNData['script1']['time'].flatten()
deltat = time[1]-time[0]
time = time - time[0]
current = DYNData['script1']['current'].flatten()
voltage = DYNData['script1']['voltage'].flatten()
soc = DYNData['script1']['soc'].flatten()

"Reserve space for predicted SOC its bounds"
sochat = np.zeros(soc.size)
socbound = np.zeros(soc.size)

"Define Covariance matrices"
SigmaX0 = np.diag(np.array([1e-6,1e-8,2e-4])) ##Initital state
SigmaV = np.array([2e-1]) ##Uncertainity of voltage sensor
SigmaW = np.array([2e-1]) ##Uncertainity of current sensor

"Call initSPKF which initializes variables using volatge and temperature measurement"
spkfData = initSPKF(voltage[0],T,SigmaX0,SigmaV,SigmaW,model)

"Loop where the SPKF is updated after each interval"
for k in range(0,len(voltage)):
    vk = voltage[k]
    ik = current[k]
    Tk = T
    sochat[k],socbound[k],spkfdata = iterSPKF(vk,ik,Tk,deltat,spkfData)


rms_error = np.sqrt(np.mean(np.square(100*(soc-sochat))))
print(rms_error)
ind = np.array(np.where(np.absolute((soc-sochat)>socbound)))
time_error = (ind.size/len(soc))*100
print(time_error)


fig1, plt1 = plt.subplots()
fig2, plt2 = plt.subplots()
fig3, (plt3,plt4,plt5) = plt.subplots(nrows=3,ncols=1,sharex=True)
plt.style.use('seaborn')

"Plot True and Estimated SOC along with error bounds"
plt1.plot(time/60,100*soc,color = 'red',label = 'True SOC',linestyle = '--')
plt1.plot(time/60,100*sochat, color = 'green',label = 'Estimated SOC')
plt1.plot(time/60,100*(sochat+socbound),time/60,100*(sochat-socbound),color = 'black',label = 'Error Bound')
plt1.set_title("SOC Estimate using SPKF")
plt1.set_xlabel("Time (min)")
plt1.set_ylabel("SOC %")
plt1.legend()

"Plot SOC Estimation error"
plt2.plot(time/60,100*(soc-sochat),color = 'red',label = 'SPKF Error')
plt2.plot(time/60,100*(socbound),time/60,100*(-socbound),color = 'black',label = 'Error Bound')
plt2.set_title("SOC Estimation error using SPKF")
plt2.set_xlabel("Time (min)")
plt2.set_ylabel("SOC Error %")
plt2.legend()

"Voltage, Current and Power plot"
plt3.plot(time/60,voltage,color = 'red',label = 'Voltage')
plt3.set_title("Voltage Vs Time")
plt3.set_ylabel("Voltage (volt)")
plt3.tick_params(axis = 'y', color = 'red')
plt3.legend()

plt4.plot(time/60,current,color = 'black',label = 'Current')
plt4.set_title("Current Vs Time")
plt4.set_ylabel("Current (amps)")
plt4.tick_params(axis = 'y', color = 'black')
plt4.legend()

plt5.plot(time/60,(voltage*current),color = 'blue',label = 'Power')
plt5.set_title("Power Vs Time")
plt5.set_xlabel("Time (min)")
plt5.set_ylabel("Power (watt)")
plt5.tick_params(axis = 'y', color = 'blue')
plt5.legend()


plt.show()