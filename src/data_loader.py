import numpy as np

def load_dataset(path, sequence_len=700, total_features=57):
    data = np.load(path)
    data = data.reshape((-1, sequence_len, total_features))
    return data

def get_data_labels(dataset, aa_residues=21, num_classes=8):
    X= dataset[:, :, 35:56]
    Y = dataset[:, :, 56:]
    mask = (np.sum(Y, axis= -1) !=0)
    return X,Y, mask