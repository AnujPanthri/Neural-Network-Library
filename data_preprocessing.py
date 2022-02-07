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
    def __init__(self,X,y,idx_to_labels=False):
        self.X=np.copy(X)
        self.y=np.copy(y)
        self.small_class=[0,np.inf]
        self.idx_to_labels=idx_to_labels
        self.data_details()
    def data_details(self):
        data_classes={"all_examples":self.y.shape[0]}
        for c in np.unique(self.y):
            data_classes[str(c)]= np.sum((self.y==c)*1)
            if np.sum((self.y==c)*1)<self.small_class[1]:
                self.small_class[0]=c
                self.small_class[1]=np.sum((self.y==c)*1)
        if self.idx_to_labels:
            new_data_classes={"all_examples":self.y.shape[0]}
            for c in data_classes:
                if c!='all_examples':
                    new_data_classes[self.idx_to_labels[int(float(c))]]=data_classes[c]
            data_classes=new_data_classes
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


def one_hot(y,classes):
    y=y.astype('int')
    one_hot=np.zeros((y.shape[0],classes))
    one_hot[np.arange(0,y.shape[0]),y.ravel()]=1
    return one_hot

# test_y=np.array([[0,1,3,2]]).T
# print(test_y.shape)
# print(one_hot(test_y,4))
