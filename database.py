import classifier
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import h5py


def make_database(dataset):
    model = classifier.Classifier()
    model.load_state_dict(torch.load('model.torch'))
    loader = torch.utils.data.DataLoader(dataset)
    features = np.empty(shape=(len(dataset), model.n_features))
    for i, (image, attr) in enumerate(loader):
        features[i] = model(image, return_features=True).detach().numpy()
    file = h5py.File('features_database.h5', 'w')
    file.create_dataset('features', features.shape, h5py.h5t.STD_U8BE, data=features)
    file.close()


if __name__ == '__main__':
    make_database(datasets.CelebA('', split='train', transform=transforms.ToTensor()))
