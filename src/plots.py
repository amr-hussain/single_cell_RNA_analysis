from pathlib import Path
import numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import roc_curve,auc
from sklearn.preprocessing import label_binarize
sns.set_theme(style='whitegrid',context='talk')

def save_embedding(Z,y,name,out):
    plt.figure(figsize=(9,7)); p=plt.scatter(Z[:,0],Z[:,1],c=y,cmap='tab10',s=8,alpha=.65); plt.colorbar(p,ticks=range(10),label='Class'); plt.title(f'{name} embedding'); plt.xlabel('Dimension 1'); plt.ylabel('Dimension 2'); plt.tight_layout(); plt.savefig(Path(out)/f'embedding_{slug(name)}.png',dpi=180); plt.close()
def save_cm(cm,name,out):
    plt.figure(figsize=(9,7)); sns.heatmap(cm,annot=True,fmt='d',cmap='Blues'); plt.title(f'{name}: confusion matrix'); plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout(); plt.savefig(Path(out)/f'cm_{slug(name)}.png',dpi=180); plt.close()
def save_roc(y,p,name,out):
    Y=label_binarize(y,classes=np.arange(10)); plt.figure(figsize=(8,7))
    for c in range(10): fpr,tpr,_=roc_curve(Y[:,c],p[:,c]); plt.plot(fpr,tpr,lw=1,label=f'{c} AUC={auc(fpr,tpr):.3f}')
    plt.plot([0,1],[0,1],'k--'); plt.xlabel('False positive rate'); plt.ylabel('True positive rate'); plt.title(f'{name}: one-vs-rest ROC'); plt.legend(ncol=2,fontsize=8); plt.tight_layout(); plt.savefig(Path(out)/f'roc_{slug(name)}.png',dpi=180); plt.close()
def save_froc(x,y,score,name,out):
    plt.figure(figsize=(8,6)); plt.plot(x,y); plt.xlabel('False-positive class decisions per image'); plt.ylabel('Sensitivity'); plt.title(f'{name}: FROC (area={score:.3f})'); plt.tight_layout(); plt.savefig(Path(out)/f'froc_{slug(name)}.png',dpi=180); plt.close()
def comparison(df,out):
    for col,title in [('accuracy','Downstream k-NN accuracy'),('cpu_seconds','CPU time (seconds)'),('auc_roc','Macro one-vs-rest ROC AUC'),('froc_auc','FROC area')]:
      plt.figure(figsize=(9,5)); sns.barplot(data=df,x='method',y=col,errorbar='sd'); plt.xticks(rotation=20); plt.title(title); plt.tight_layout(); plt.savefig(Path(out)/f'comparison_{col}.png',dpi=180); plt.close()
def slug(s): return s.lower().replace(' + ','_').replace('-','').replace(' ','_')
