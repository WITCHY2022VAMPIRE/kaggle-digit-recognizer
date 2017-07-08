from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import itertools

def pca_reduction(features, RATIO_EXP= 0.95):
    
    
    pca_model= PCA()
    pca_model.fit(features)
    
#    accVR= list( itertools.accumulate(pca_model.explained_variance_ratio_) )
#    plt.plot(pca_model.explained_variance_ratio_)
#    plt.plot(accVR)
#    plt.show()

    acc= 0.; n_rf= 0 # N of reduced features
    for i, r in enumerate(pca_model.explained_variance_ratio_):
        acc += r
        if acc > RATIO_EXP: n_rf= i+1; break
    
    return pca_model.transform(features)[:n_rf]
    
