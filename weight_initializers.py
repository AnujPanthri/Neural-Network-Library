import numpy as np

def random_initializer(shape):
    nodes=np.prod(shape)
    w=np.random.randn(nodes).reshape(shape)
    return w
def he_initializer(shape):
    nodes=np.prod(shape)
    prev=shape[0]
    w=np.random.randn(nodes).reshape(shape)*np.sqrt(2/prev)
    return w