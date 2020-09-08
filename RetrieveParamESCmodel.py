import numpy as np
from scipy import interpolate


"This part of the code returns values of ESC cell parameter 'paramName' for the specified temperature"
"The parameters in 'paramName' can be any of the following 'QParam', 'RCParam', 'R0Param', 'MParam', 'M0Param', 'etaParam', 'GParam'"

def getParamESC(paramName,temp,model):
    theFields = list(model) ##Get parameter names from ESC model
    match = theFields.index(paramName) ##Check if the paramName matches any name from the list an return the indice of the same
    fieldName = theFields[match]

    if np.isscalar(model['temps']):## If a temp variable is scallar i.e. data for only one temperature is available
        if temp in model['temps']:
            theParam = model[fieldName]
            return theParam
        else:## If the data for the requested temp is not available
            print("Model does not contain requested data at this temperature")

    ## Data for multiple temp is available
    theParamData = model[fieldName]
    temp = max(min(temp,max(model['temps'])),min(model['temps']))
    ind = np.array(np.where(model['temps']==temp)) ## Check if the data for exact temp is availble, helps in avoiding unnecessary interpolation
    if ind.size != 0:
        if theParamData.size == 1:
            theParam = theParamData[ind]
        else:
            theParam = theParamData[ind]
    else: ## If data at exact temp is not available
        tck = interpolate.splrep(model['temps'], theParamData,s=0)
        theParam = interpolate.splev(temp,tck,der=0)

    return theParam











