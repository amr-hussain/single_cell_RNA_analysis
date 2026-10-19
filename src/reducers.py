import time, numpy as np
from sklearn.decomposition import PCA,KernelPCA
from sklearn.manifold import TSNE
from .sne import GaussianSNE

def timed(name,fn):
    t0=time.process_time(); w0=time.perf_counter(); Z,obj=fn()
    return {'method':name,'embedding':Z,'model':obj,'cpu_seconds':time.process_time()-t0,'wall_seconds':time.perf_counter()-w0}

def run_reducers(X,cfg,seed):
    out=[]
    out.append(timed('PCA',lambda:(lambda m:(m.fit_transform(X),m))(PCA(n_components=2,random_state=seed))))
    out.append(timed('Kernel PCA',lambda:(lambda m:(m.fit_transform(X),m))(KernelPCA(n_components=2,kernel='rbf',gamma=cfg['kpca_gamma'],eigen_solver='randomized',random_state=seed,fit_inverse_transform=True))))
    # Exact SNE is deliberately capped because it forms dense n×n matrices.
    n_sne=min(len(X),2000); idx=np.random.default_rng(seed).choice(len(X),n_sne,replace=False)
    sne=GaussianSNE(2,cfg['sne_perplexity'],cfg['sne_iterations'],random_state=seed)
    r=timed('SNE',lambda:(sne.fit_transform(X[idx]),sne)); r['subset_index']=idx; out.append(r)
    ts=TSNE(n_components=2,perplexity=cfg['tsne_perplexity'],max_iter=cfg['tsne_iterations'],init='pca',learning_rate='auto',random_state=seed)
    out.append(timed('t-SNE',lambda:(ts.fit_transform(X),ts)))
    p=PCA(n_components=min(cfg['pca_precomponents'],X.shape[1]),random_state=seed); Xp=p.fit_transform(X)
    ht=TSNE(n_components=2,perplexity=cfg['tsne_perplexity'],max_iter=cfg['tsne_iterations'],init='pca',learning_rate='auto',random_state=seed)
    out.append(timed('PCA + t-SNE',lambda:(ht.fit_transform(Xp),ht)))
    return out
