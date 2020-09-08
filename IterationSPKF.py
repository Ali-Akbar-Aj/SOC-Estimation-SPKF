import numpy as np
import scipy.io as sio
from SOCfromOCVtemp import SOCfromOCVtemp
from OCVfromSOCtemp import OCVfromSOCtemp
from RetrieveParamESCmodel import getParamESC
from InitializeSPKF import initSPKF

"This function performs one round of iteration from the new measured data"
"Inputs to the function are voltage, current, temperature, sampling interval "
"and data structure initialized by initSPKF and updated by iterSPKF"
def iterSPKF(vk, ik, Tk, deltat, spkfData):
    ## Load cell parameters from ESC model
    model = spkfData['model']
    Q = getParamESC('QParam',Tk,model)
    G = getParamESC('GParam',Tk,model)
    M = getParamESC('MParam',Tk,model)
    M0 = getParamESC('M0Param',Tk,model)
    RC = np.exp(np.divide(-deltat,(np.absolute(getParamESC('RCParam',Tk,model)))))
    R = getParamESC('RParam', Tk, model)
    R0 = getParamESC('R0Param', Tk, model)
    eta = getParamESC('etaParam', Tk, model)

    if ik<0:## for charging current take the efficiency factor into account
        ik = np.multiply(ik,eta)

    ## Retrieve data from spkfdata structure
    I = spkfData['priorI']
    SigmaX = spkfData['SigmaX']
    xhat = spkfData['xhat']
    Nx = spkfData['Nx']
    Nw = spkfData['Nw']
    Nv = spkfData['Nv']
    Na = spkfData['Na']
    Snoise = spkfData['Snoise']
    Wc = spkfData['Wc']
    irInd = spkfData['irInd']
    hkInd = spkfData['hkInd']
    zkInd = spkfData['zkInd']
    if np.absolute(ik) > (Q/100):
        spkfData['signIk'] = np.sign(ik)
    signIk = spkfData['signIk']

    ##Step 1a:  State estimate time update

    ##Step 1a-1: Create SigmaX and xhat augmented matrix
    try:
        sigmaXa = np.linalg.cholesky(SigmaX)
    except:
        theAbsDiag = np.absolute(np.diag(SigmaX))
        sigmaXa = np.diag(np.max(np.sqrt(theAbsDiag),np.sqrt(spkfData['SigmaW']))) ## Lower triangular matrix

    ##Step 1a-2: Calculate SigmaX points
    a = np.real(sigmaXa)
    b = np.zeros((Nx,(Nw+Nv)))
    c = np.concatenate((a,b),axis=1)
    d = np.zeros(((Nw+Nv),Nx))
    e = np.concatenate((d,Snoise),axis=1)
    sigmaXa = np.concatenate((c,e))
    ir0 = spkfData['xhat'][0]
    hk0 = spkfData['xhat'][1]
    SOC0 = spkfData['xhat'][2]
    xhat = np.array((ir0, hk0, SOC0)).reshape((3,1))
    zeros = np.zeros(((Nw+Nv),1))
    xhata = np.concatenate((xhat,zeros))
    xhatarepeat = np.tile(xhata,(2*Na+1))
    f = np.zeros((Na,1))
    g = np.concatenate((f,sigmaXa),axis=1)
    j = np.concatenate((g,-sigmaXa),axis=1)
    Xa = xhatarepeat + spkfData['h']*j

    ## Calculate new states for all the old state vectors
    def stateEqn(xold,current,xnoise):
        current = current + xnoise
        xnew = 0*xold
        k = RC * xold[irInd, :]
        xnew[irInd,:] = RC*xold[irInd,:]+(1-RC)*current
        Ah = np.exp(-1*np.absolute((current*G*deltat)/(3600*Q)))
        xnew[hkInd,:] = np.multiply(Ah,xold[hkInd,:]) + np.multiply((Ah-1),np.sign(current))
        xnew[zkInd,:] = xold[zkInd,:] - (current*deltat)/(3600*Q)
        return xnew

    ##Step 1a-3: Time update from last iteration
    Xx = stateEqn(Xa[0:Nx,:],I,Xa[Nx:Nx+Nw,:])
    xhat = np.dot(Xx,spkfData['Wm']).reshape((3,1))
    xhat[hkInd] = min(1,max(-1,xhat[hkInd]))
    xhat[zkInd] = min(1.05,max(-0.05,xhat[zkInd]))

    ##Step 1b: Error covariance time update
    n = np.tile(xhat,(2*Na+1))
    Xs = Xx - n
    SigmaX = np.dot((np.dot(Xs,np.diag(Wc))),np.transpose(Xs))

    ##Step 1c: Output estimate
    I = ik
    yk = vk

    ##Calculate cell output volatge for all state vectors in xhat
    def outputEqn(xhat,current,ynoise,T,model):
        yhat = OCVfromSOCtemp(xhat[zkInd,:],T,model)
        yhat = yhat + M*xhat[hkInd,:] + M0*signIk
        yhat = yhat - R * xhat[irInd, :] - R0 * current + ynoise
        return yhat

     ##Step 2a: Estimator gain matrix
    Y = outputEqn(Xx, I+Xa[Nx:Nx+Nw,:], Xa[Nx+Nw,:],Tk,model)
    yhat = np.dot(Y,spkfData['Wm'])

    p = np.tile(yhat, (2*Na+1))
    Ys = Y - p
    SigmaXY = np.dot((np.dot(Xs, np.diag(Wc))), np.transpose(Ys))
    SigmaY = np.dot((np.dot(Ys, np.diag(Wc))), np.transpose(Ys))
    L = np.divide(SigmaXY,SigmaY)

    ##Step 2b: State estimate measurement update
    r = yk-yhat

    if r*r >100:
        L[:,0] = 0
    xhat = xhat + L*r
    xhat[zkInd] = min(1.05, max(-0.05, xhat[zkInd]))

    ##Step 2c: Error covariance measurement update
    SigmaX = SigmaX - L*SigmaY*np.transpose(L)
    u, S, V = np.linalg.svd(SigmaX, full_matrices=True)
    S = np.diag(S)
    V = np.transpose(V)
    HH = np.dot((np.dot(V,S)),np.transpose(V))
    SigmaX = (SigmaX +np.transpose(SigmaX) + HH + np.transpose(HH))/4

    ## Q-bump
    if r*r>4*SigmaY:
        SigmaX[zkInd,zkInd] = SigmaX[zkInd,zkInd]*spkfData['Qbump']

    ## Save data in spkfdata structure for the next itteration
    spkfData['priorI'] = ik
    spkfData['SigmaX'] = SigmaX
    spkfData['xhat'] = xhat
    zk = xhat[zkInd]
    zkbnd = 3*np.sqrt(SigmaX[zkInd,zkInd])

    return zk,zkbnd,spkfData



