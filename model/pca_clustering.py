import torch
from sklearn.decomposition import PCA
from sklearn import preprocessing

import pickle
import joblib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v,f)
    f.close()
    return filename

def load_variable(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r

def get_en_out(data_loader, model):
    """
    Get the encoder outputs of the given data from the model
    """
    flag = 1

    for idx, data in enumerate(data_loader):
        x=data.x.cuda()   
        edge_index = data.edge_index.cuda()
        edge_attr = data.edge_attr.cuda()
        y=data.y.cuda()

        cls_, reg_, en_cls_out, en_reg_out = model(x,edge_index,edge_attr)#.to(device)
        en_out = torch.cat((en_cls_out,en_reg_out),-1)

        if flag:
          en_outs = en_out
          flag = 0
        else:
          en_outs = torch.cat((en_outs,en_out),0)

    return en_outs

def vis_pca_heatmap(pca_model, en_outs):
    """"
    visualize pca heatmap
    """
    X_new = pca_model.transform(en_outs)
    x = X_new[:,0]
    y = X_new[:,1]

    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    z = preprocessing.maxabs_scale(z,axis=0, copy=True)

    idx1 = z.argsort()
    x2, y2, z2 = x[idx1], y[idx1], z[idx1]

    fig2, ax2 = plt.subplots()

    ax2.set_xlabel('PCA_dim1', fontsize=10)
    ax2.set_ylabel('PCA_dim2', fontsize=10)

    img = ax2.scatter(x2, y2, c=z2, s=0.5, edgecolors="none",cmap="inferno")#cmap='Reds')
    cbar = plt.colorbar(img, ax=ax2)

    plt.grid()

    plt.show()

def save_pca_model(en_outs, dim, pca_dir):
    pca = PCA(n_components=dim)
    pca.fit(en_outs)
    joblib.dump(pca, pca_dir+'pca.m')

def load_pca_model(pca_dir):
    pca_model = joblib.load(pca_dir)

    return pca_model