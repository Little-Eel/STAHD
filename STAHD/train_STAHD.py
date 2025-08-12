import numpy as np
import pandas as pd
from statsmodels.genmod.tests.results.glm_test_resids import cpunish
from tqdm import tqdm
import scipy.sparse as sp

from .STAHD import STAHD

import torch
import torch.backends.cudnn as cudnn

cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborLoader

def train_STAHD(adata, hidden_dims=[512, 30], n_epochs=1000, lr=0.001, key_added='STAHD',
                    gradient_clipping=5., weight_decay=0.0001, margin=1.0, verbose=False,
                    random_seed=666, iter_comb=None, knn_neigh=100, batch_size = 256,
                    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    """\
    Train graph attention auto-encoder and use spot triplets across slices to perform batch correction in the embedding space.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    hidden_dims
        The dimension of the encoder.
    n_epochs
        Number of total epochs in training.
    lr
        Learning rate for AdamOptimizer.
    key_added
        The latent embeddings are saved in adata.obsm[key_added].
    gradient_clipping
        Gradient Clipping.
    weight_decay
        Weight decay for AdamOptimizer.
    margin
        Margin is used in triplet loss to enforce the distance between positive and negative pairs.
        Larger values result in more aggressive correction.
    iter_comb
        For multiple slices integration, we perform iterative pairwise integration. iter_comb is used to specify the order of integration.
        For example, (0, 1) means slice 0 will be algined with slice 1 as reference.
    knn_neigh
        The number of nearest neighbors when constructing MNNs. If knn_neigh>1, points in one slice may have multiple MNN points in another slice.
    device
        See torch.device.

    Returns
    -------
    AnnData
    """

    # seed_everything()
    seed = random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    edgeList = adata.uns['edgeList']
    data = Data(x=torch.FloatTensor(adata.X.todense()),
                edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]])))
    # data = data.to(device)


    cluster_data = ClusterData(data, num_parts=int(np.ceil(data.num_nodes / batch_size)) * 10,
                               recursive=False, log=False)
    train_loader = ClusterLoader(cluster_data, batch_size=10, shuffle=True)
    subgraph_loader = NeighborLoader(data, num_neighbors=[-1], batch_size=batch_size,
                                     shuffle=False)

    model = STAHD(hidden_dims=[data.x.shape[1], hidden_dims[0], hidden_dims[1]]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if verbose:
        print(model)

    print('Pretrain with STAHD...')
    for epoch in tqdm(range(0, n_epochs)):
        for batch in train_loader:
            model.train()
            optimizer.zero_grad()
            batch.to(device)
            z, out = model(batch.x, batch.edge_index)

            loss = F.mse_loss(batch.x, out)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()

    with torch.no_grad():
        z_list = []
        out_list = []

        #device_test = cpu
        device_test = torch.device('cpu')
        for batch in subgraph_loader:
            batch.to(device_test)
            model.to(device_test)
            z, out = model(batch.x, batch.edge_index)
            z_list.append(z[:batch.batch_size].cpu())
            out_list.append(out[:batch.batch_size].cpu())

        z_all = torch.cat(z_list, dim=0)
        out_all = torch.cat(out_list, dim=0)
    adata.obsm['STAHD'] = z_all.numpy()
    ReX = out_all.detach().numpy()
    ReX[ReX < 0] = 0
    adata.layers['STAHD_ReX'] = ReX

    return adata

