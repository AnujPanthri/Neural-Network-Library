import numpy as np


class polynomial:
    def __init__(self,order=1):
        self.order=order
    def forward(self,test_x):
        num_of_features=test_x.shape[-1]
        for i in range(2,self.order+1):
            test_x=np.concatenate([test_x,test_x[:,0:num_of_features]**i],axis=-1)
        return test_x




class feature_scaling:
    ''' intialize with X data 
        and use transform(test_x) to scale the data 
        and transform_to_normal(test_x) to get back original data'''
    def __init__(self,X):
        self.mu=np.average(X,axis=0)
        self.sigma=np.std(X,axis=0,ddof=1)
        self.mu=np.expand_dims(self.mu,axis=0)
        self.sigma=np.expand_dims(self.sigma,axis=0)
    def transform(self,test_x):
        # print(mu.shape)
        # print(sigma.shape)
        test_x=(test_x-self.mu)/self.sigma
        # print(test_x.shape)
        return test_x
    def transform_to_normal(self,test_x):
        test_x=(test_x*self.sigma)+self.mu
        return test_x




class undersampling:
    def __init__(self,X,y):
        self.X=np.copy(X)
        self.y=np.copy(y)
        self.small_class=[0,np.inf]
        self.data_details()
    def data_details(self):
        data_classes={"all_examples":self.y.shape[0]}
        for c in np.unique(self.y):
            data_classes[str(c)]= np.sum((self.y==c)*1)
            if np.sum((self.y==c)*1)<self.small_class[1]:
                self.small_class[0]=c
                self.small_class[1]=np.sum((self.y==c)*1)
        print(data_classes)
        return data_classes
    def get_data(self):
        # self.small_class[1]=2
        X=np.array([[]])
        y=np.array([[]])
        for c in np.sort(np.unique(self.y)):
            idx,bool=np.where(self.y==c)
            # print(c)
            # print(idx)
            # print(bool)
            np.random.shuffle(idx)
            # print(idx.shape)
            idx=idx[:self.small_class[1]]
            # print(idx.shape)
            if c==0:
                X=self.X[idx,:]
                y=self.y[idx,:]
            else:
                X=np.r_[X,self.X[idx,:]]
                y=np.r_[y,self.y[idx,:]]
        return X,y


