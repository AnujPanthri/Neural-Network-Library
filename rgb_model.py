import dynamic_nn as dn
import numpy as np
import matplotlib.pyplot as plt
from data import *
from helper import *
from confusion_matrix import *
from gradient_checking import *

model=dn.nn()
# model=dn.nn(use_bias="False")
model.add(dn.input_layer((3)))

model.add(dn.dense(20,activation='sigmoid'))
model.add(dn.dense(20,activation='sigmoid'))


model.add(dn.dense(11,activation='softmax',last_layer=True))
model.make_model()
model.summary()
model.compile(loss=dn.categorical_crossentropy(),metrics=[categorical_acc],metrics_names=['acc'])

# if training is unstable try reducing lr ,or if training slow try increasing the lr

# sigmoid
model.lr=0.3  # sigmoid
model.lr=0.5  # sigmoid
model.lr=0.8  # sigmoid # best lr 2 hidden layers with 20 nodes
# model.lr=1  # sigmoid  # slight overfitting
# model.lr=1.5  # sigmoid  # slight overfitting unstable training




''' Experiment with these data also , note you might have to adjust lr(learning rate) according to each data '''

X,y,labels = load_rgb() # data loading
X=X/255
idx_to_labels={i:label for i,label in enumerate(labels)}

from data_preprocessing import *
u_s=undersampling(X,y,idx_to_labels)
input("press enter to continue....")

y=dn.data_preprocessing.one_hot(y,classes=len(labels))

print('X:',X.shape)
print('y:',y.shape)

# print(y[-5:,:])

x_train,y_train,x_test,y_test=split_data(X,y)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)


# gradient_checking(model,X,y) # gradient checking before training
# model.fit(x_train,y_train,iter=500,verbose=0)
# gradient_checking(model,X,y) # gradient checking after training


# input("press enter to continue....")
# cost_lr,grad_lr=model.lr_finder(x_train,y_train,0.0001,2,100)
# model.lr=grad_lr
# model.lr=cost_lr

# model.learning_curve(x_train,y_train,(x_test,y_test),6)

model.fit(x_train,y_train,val_data=(x_test,y_test),iter=5000,verbose=1)
# model.fit(x_train,y_train,val_data=(x_test,y_test),iter=10000,verbose=1)


plot=plotter(graph_name="loss",y_axis="loss")
plot.add(model.cost_hist,y_name="train_loss")
plot.add(model.val_cost_hist,y_name="val_loss")
plot.show()

plot=plotter(graph_name="accuracy",y_axis="acc")
plot.add(model.metrics_hist[0],y_name="train_acc")
plot.add(model.val_metrics_hist[0],y_name="val_acc")
plot.show()



y_pred_train=model.predict(x_train)
y_pred_test=model.predict(x_test)

print("Train acc:",categorical_acc(y_train,y_pred_train))
print("Test acc:",categorical_acc(y_test,y_pred_test))




y_test=y_test.astype("int")
y_test=np.argmax(y_test,axis=-1)[...,None]
y_pred_test=(model.predict(x_test)>=0.5)*1
y_pred_test=np.argmax(y_pred_test,axis=-1)[...,None]

print(y_test.shape,y_pred_test.shape)
con=confusion(y_test,y_pred_test,classes=len(labels))
print("confusion_matrix:\n",con)
show_confusion_matrix(con,labels)
f1score=f1
all=f1score(con)
print("Names:",all[0])
print("Precision:",all[1])
print("Recall:",all[2])
print("F1:",all[3])
print("Support:",all[4])





print(x_test.shape)
pixels=8
as_image=np.tile(x_test,(pixels**2)).reshape(x_test.shape[0],pixels,pixels,3)
as_image*=255
as_image=as_image.astype(np.uint8)
print(as_image.shape)
out=y_pred_test
# print(as_image)
fig=plt.figure()
ex=50
col=8
import math
plt.title("Predictions")
plt.axis('off')
for i in range(ex):
    fig.add_subplot(math.ceil(ex/col),col,(i+1))
    # plt.title(f"{out[i,0]}")
    plt.title(f"{idx_to_labels[out[i,0]]}")
    plt.imshow(as_image[i,:],cmap='gray')
    plt.axis('off')
fig.subplots_adjust(hspace=0.433)
fig.tight_layout()
plt.show()
