import classifier
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import h5py
import faiss
import time


def make_database_h5py(dataset):
    model = classifier.Classifier().cuda()
    model.load_state_dict(torch.load('model.torch'))
    model.eval()
    loader = torch.utils.data.DataLoader(dataset)
    features = np.empty(shape=(len(dataset), model.n_features))
    attributes = np.empty(shape=(len(dataset), 40))
    start = time.time()
    with torch.no_grad():
        for i, (image, attr) in enumerate(loader):
            if i > 0 and i % int(len(loader) / 100) == 0:
                print(f'Loaded {i} images')
            attributes[i] = attr.numpy()
            features[i] = model(image.cuda(), return_features=True).cpu().numpy()
    with h5py.File('features_database.h5', 'w') as file:
        file.create_dataset('attributes', attributes.shape, data=attributes)
        file.create_dataset('features', features.shape, h5py.h5t.STD_U8BE, data=features)
    print(f'Done in {time.time() - start:.2f}s')


def make_database(dataset):
    model = classifier.Classifier().cuda()
    model.load_state_dict(torch.load('model.torch'))
    model.eval()
    loader = torch.utils.data.DataLoader(dataset)
    features = np.empty(shape=(len(dataset), model.n_features), dtype=np.float32)
    start = time.time()
    with torch.no_grad():
        for i, (image, _) in enumerate(loader):
            if i < 10:
                continue
            if i > 0 and i % int(len(loader) / 100) == 0:
                print(f'Loaded {i} images')
            features[i] = model(image.cuda(), return_features=True).cpu().numpy()
    index = faiss.IndexFlatL2(model.n_features)
    # noinspection PyArgumentList
    index.add(features)
    faiss.write_index(index, 'feature_index.faiss')
    print(f'Done in {time.time() - start:.2f}s')


if __name__ == '__main__':
    make_database(datasets.CelebA('', split='train', transform=transforms.ToTensor()))
