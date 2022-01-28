import dynamic_nn as dn
import numpy as np
import matplotlib.pyplot as plt
from data import *
from helper import *

model=dn.nn()
# model=dn.nn(use_bias="False")
model.add(dn.input_layer((2)))
model.add(dn.dense(5,activation='sigmoid'))
model.add(dn.dense(1,activation='sigmoid',last_layer=True))
model.make_model()
model.summary()

# if training is unstable try reducing lr ,or if training slow try increasing the lr

# model.lr=0.4
# model.lr=2 
model.lr=1

''' Experiment with these data also , note you might have to adjust lr(learning rate) according to each data '''
# X,y = load_planar_dataset() # data loading
# X,y = load_moon() # data loading         
X,y = load_circle() # data loading
# X,y = load_blob() # data loading

print('X:',X.shape)
print('y:',y.shape)



show(model,X,y)
model.fit(X,y,iter=5000)
model.plot_training() # plot training graph (loss vs number of iterations). if this graph is not smooth training is unstable
show(model,X,y)