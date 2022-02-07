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
    data=np.loadtxt("data/moon.txt")
    X,y=data[:,:2],data[:,2:3]
    return X,y

def load_blob():
    data=np.loadtxt("data/blob.txt")
    X,y=data[:,:2],data[:,2:3]
    return X,y
    
def load_circle():
    data=np.loadtxt("data/circle.txt")
    X,y=data[:,:2],data[:,2:3]
    return X,y


import pandas as pd
def load_heart_disease():
    data=pd.read_csv('data/framingham.csv')
    Missing_values_percent = 100*(data.isnull().sum()/len(data["male"]))
    data["education"].fillna(data["education"].median(), inplace = True)
    data["cigsPerDay"].fillna(data["cigsPerDay"].median(), inplace = True)
    data["BPMeds"].fillna(data["BPMeds"].median(), inplace = True)
    data["totChol"].fillna(data["totChol"].median(), inplace = True)
    data["BMI"].fillna(data["BMI"].median(), inplace = True)
    data["heartRate"].fillna(data["heartRate"].median(), inplace = True)
    data["glucose"].fillna(data["glucose"].median(), inplace = True)
    Missing_values_percent = 100*(data.isnull().sum()/len(data["male"]))
    # print(Missing_values_percent)
    X = np.array(data.drop(["TenYearCHD"], axis = 1))
    y = np.array(data["TenYearCHD"]) # target
    y=y.reshape(-1,1)
    return X,y


def load_rgb():
    data=pd.read_csv('data/rgb.csv')
    # data=np.array(data)
    data=data.to_numpy()
    # print(data)
    labels=np.unique(data[:,-1])
    # print(labels)
    for i,label in enumerate(labels):
        data[(data==label)]=i
    # print(data)
    data=data.astype(np.float32)
    x,y=data[:,:-1],data[:,-1:]
    # print(x.shape,y.shape)
    return x,y,labels
