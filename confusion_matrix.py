import numpy as np
def confusion(y_true,y_pred,classes):
    c_matrix=np.zeros((classes,classes))
    for i in range(y_true.shape[0]):
        c_matrix[int(y_pred[i]),int(y_true[i])]+=1
        # c_matrix[y_pred[i,0],y_true[i,0]]+=1
    return c_matrix
def show_confusion_matrix(con,labels):
    import matplotlib.pyplot as plt
    figure = plt.figure()
    axes = figure.add_subplot(111)
    caxes = axes.matshow(con,cmap=plt.get_cmap('Blues'))
    # caxes = axes.matshow(con,cmap=plt.get_cmap('cool'))
    figure.colorbar(caxes)
    
    axes.xaxis.set_ticks(np.arange(0,con.shape[0],1))
    axes.yaxis.set_ticks(np.arange(0,con.shape[0],1))
    axes.set_xticklabels(labels)
    axes.set_yticklabels(labels)

    for (i, j), z in np.ndenumerate(con):
        axes.text(j, i, int(z), ha='center', va='center')
    plt.show()
    # plt.figure()
    # plt.matshow(con,'Blues')
    # ax=plt.matshow(con,cmap=plt.get_cmap('Blues'))
    # plt.colorbar()
    # plt.show()
def f1(confusion_matrix):
    precision=np.zeros(confusion_matrix.shape[1])
    recall=np.zeros(confusion_matrix.shape[1])
    f1=np.zeros(confusion_matrix.shape[1])
    support=np.zeros(confusion_matrix.shape[1])
    for c in range(confusion_matrix.shape[1]):
        precision[c]=confusion_matrix[c,c]/np.sum(confusion_matrix[c,:]) # true positives / predicted positives
        recall[c]=confusion_matrix[c,c]/np.sum(confusion_matrix[:,c]) # true positives / actual positives
        f1[c]=(2*precision[c]*recall[c])/(precision[c]+recall[c]) # 2*P*R/P+R
        support[c]=confusion_matrix[c,c] # number of true positives in that class

    # print("Precision:",precision)
    # print("Recall:",recall)
    # print("F1:",f1)
    # print("Support:",support)
    # all={"Precision:",precision,"Recall:",recall,"F1:",f1,"Support:",support}
    names={"Precision","Recall","F1","Support"}
    return names,precision,recall,f1,support


if __name__=='__main__':
    # y_true=np.array([[1,0,1,0,1,1,1,0,0]]).T
    # y_pred=np.array([[1,0,0,0,0,0,1,1,0]]).T
    y_true=np.array([[1,2,1,0,0,1]]).T
    y_pred=np.array([[1,2,0,1,0,1]]).T

    c_m=confusion(y_true,y_pred)
    print("confusion_matrix:\n",c_m)
    all=f1(c_m)

    print("Precision:",all[0])
    print("Recall:",all[1])
    print("F1:",all[2])
