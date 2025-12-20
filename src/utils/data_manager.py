import numpy as np

def partition_data(labels, num_clients, alpha=0.5):
    """
    Partitions data using Dirichlet distribution for Non-IID simulation.
    """
    num_classes = len(np.unique(labels))
    
    label_distribution = np.random.dirichlet([alpha] * num_clients, num_classes)
    
    client_idcs = [[] for _ in range(num_clients)]
    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        
        proportions = label_distribution[k]
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        split_indices = np.split(idx_k, proportions)
        
        for i, idx in enumerate(split_indices):
            client_idcs[i].append(idx)

    return [np.concatenate(idcs) for idcs in client_idcs]