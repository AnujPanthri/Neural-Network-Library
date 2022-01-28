import numpy as np

def load_planar_dataset():
    np.random.seed(1)
    m = 400 # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
        
    return X, Y


def load_moon():
    data=np.loadtxt("moon.txt")
    X,y=data[:,:2],data[:,2:3]
    return X,y

def load_blob():
    data=np.loadtxt("blob.txt")
    X,y=data[:,:2],data[:,2:3]
    return X,y
    
def load_circle():
    data=np.loadtxt("circle.txt")
    X,y=data[:,:2],data[:,2:3]
    return X,y