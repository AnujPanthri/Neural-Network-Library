import numpy as np
from activations import *


def ReLU_function( signal, derivative=False ):
    if derivative:
        return (signal > 0).astype(float)
    else:
        # Return the activation signal
        return np.maximum( 0, signal )


data=np.array([23,25,-34,0])
print("input data:",data)
act=relu()
print("output:",act.forward(data))
print("derivatives:",act.backward(data))



print("output:",ReLU_function(data))
print("derivatives:",ReLU_function(data,True))

