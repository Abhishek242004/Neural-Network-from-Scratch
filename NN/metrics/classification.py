import numpy as np

class Accuracy:
    def __call__(self,y_pred,y):
        y_pred_label=np.argmax(y_pred)
        y_label=np.argmax(y)
        return np.mean(y_pred_label==y_label)

class ConfusionMatrix:
    def __call__(self,y_pred,y):
        y_pred_label=np.argmax(y_pred)
        y_label=np.argmax(y)

        num_classes=y.shape[1]
        cm=np.zeros((num_classes,num_classes),dtype=int)

        for t,p in zip(y_label,y_pred_label):
            cm[t,p]+=1
        return cm 
    
class Precision:
    def __call__(self,y_pred,y):
        cm=ConfusionMatrix()(y_pred,y)
        precision=np.diag(cm)/(np.sum(cm,axis=0)+1e-8)
        return precision
class Recall:
    def __call__(self,y_pred,y):
        cm=ConfusionMatrix()(y_pred,y)
        recall=np.diag(cm)/(np.sum(cm,axis=1)+1e-8)
        return recall

class F1:
    def __call__(self,y_pred,y):
        precision=Precision()(y_pred,y)
        recall=Recall()(y_pred,y)
        f1_score=(precision*recall)/(precision+recall)
        return f1_score

