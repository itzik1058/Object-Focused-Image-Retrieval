import classifier
import torch
import torch.nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time


def train(n_epoch):
    transform = transforms.Compose([transforms.Resize(64), transforms.CenterCrop(64),
                                    transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_dataset = datasets.CelebA('', split='train', transform=transform)
    # train_dataset = torch.utils.data.Subset(train_dataset, range(10000))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    model = classifier.Classifier().cuda()
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.BCELoss()
    start = time.time()
    for epoch in range(n_epoch):
        epoch_loss, epoch_accuracy = 0, 0
        epoch_start = time.time()
        for image, true_attr in train_loader:
            image, true_attr = image.cuda(), true_attr.cuda()
            attr = torch.sigmoid(model(image))
            loss = loss_fn(attr, true_attr.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() / len(train_dataset)
            epoch_accuracy += attr.ge(0.5).eq(true_attr).sum() / (40 * len(train_dataset))
        print(f'Epoch {epoch:02} with loss {epoch_loss:.5f}, accuracy {epoch_accuracy:.3f} '
              f'took {time.time() - epoch_start:.2f}s')
    print(f'Done in {time.time() - start:.2f}s')
    torch.save(model.state_dict(), 'model.torch')


if __name__ == '__main__':
    train(n_epoch=5)
