from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import itertools

def pca_reduction(features, RATIO_EXP= 0.95):

    pca_model= PCA()
    f_transformed= pca_model.fit_transform(features)
    
#    accVR= list( itertools.accumulate(pca_model.explained_variance_ratio_) )
#    plt.plot(pca_model.explained_variance_ratio_)
#    plt.plot(accVR)
#    plt.show()

    acc= 0.; n_rf= 0 # N of reduced features
    for i, r in enumerate(pca_model.explained_variance_ratio_):
        acc += r
        if acc > RATIO_EXP: n_rf= i+1; break
   
    print("Reduce size to:", n_rf)
    return f_transformed[:,:n_rf], pca_model, n_rf
#    return f_transformed
    
def pca_transform(features, pca_model, n_rf):
    return pca_model.transform(features)[:,:n_rf]
