import numpy as np, pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve,cohen_kappa_score,accuracy_score,precision_recall_fscore_support
from sklearn.preprocessing import label_binarize

def evaluate(method,Z,y,train_mask,k=7):
    clf=KNeighborsClassifier(k,weights='distance').fit(Z[train_mask],y[train_mask])
    pred=clf.predict(Z[~train_mask]); proba=clf.predict_proba(Z[~train_mask]); yt=y[~train_mask]
    cm=confusion_matrix(yt,pred,labels=np.arange(10)); rows=[]
    for c in range(10):
        tp=cm[c,c]; fn=cm[c].sum()-tp; fp=cm[:,c].sum()-tp; tn=cm.sum()-tp-fn-fp
        sens=tp/(tp+fn+1e-12); spec=tn/(tn+fp+1e-12); prec=tp/(tp+fp+1e-12); npv=tn/(tn+fn+1e-12)
        rows.append(dict(method=method,class_id=c,sensitivity=sens,recall=sens,specificity=spec,precision=prec,f1=2*prec*sens/(prec+sens+1e-12),informedness=sens+spec-1,markedness=prec+npv-1))
    Y=label_binarize(yt,classes=np.arange(10)); auc=roc_auc_score(Y,proba,multi_class='ovr',average='macro')
    acc=accuracy_score(yt,pred)
    summary=dict(method=method,accuracy=acc,error=1-acc,macro_sensitivity=np.mean([r['sensitivity'] for r in rows]),macro_specificity=np.mean([r['specificity'] for r in rows]),macro_precision=np.mean([r['precision'] for r in rows]),macro_recall=np.mean([r['recall'] for r in rows]),macro_f1=np.mean([r['f1'] for r in rows]),auc_roc=auc,kappa=cohen_kappa_score(yt,pred),informedness=np.mean([r['informedness'] for r in rows]),markedness=np.mean([r['markedness'] for r in rows]))
    return summary,pd.DataFrame(rows),cm,yt,pred,proba,clf

def froc_curve(y_true,proba,thresholds=101):
    # Multiclass one-vs-rest image-level FROC: sensitivity vs mean false-positive class decisions per image.
    Y=label_binarize(y_true,classes=np.arange(proba.shape[1])); xs=[]; ys=[]
    for t in np.linspace(0,1,thresholds):
        call=proba>=t; tp=(call & (Y==1)).sum(); pos=(Y==1).sum(); fp=(call & (Y==0)).sum()
        ys.append(tp/(pos+1e-12)); xs.append(fp/len(y_true))
    order=np.argsort(xs); x=np.array(xs)[order]; y=np.array(ys)[order]
    return x,y,float(np.trapezoid(y,x))
