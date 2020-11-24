import torch

from sklearn.decomposition import PCA
from torchvision import datasets, transforms
from torchvision.utils import save_image

n_components = 16


def main():
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), shuffle=True)
    test_set = datasets.MNIST('./data', train=False, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=True)
    for x, y in train_loader:
        X_train = x.view(-1, 784).numpy()
        y_train = y.numpy()
    for x, y in test_loader:
        X_test = x.view(-1, 784).numpy()
        y_test = y.numpy()
    pca = PCA(n_components=n_components)
    pca.fit_transform(X_train)
    X_test_reduced = pca.transform(X_test)
    X_test_recovered = pca.inverse_transform(X_test_reduced)
    comparison = torch.cat([torch.from_numpy(X_test[:8]).view(-1, 1, 28, 28),
                            torch.from_numpy(X_test_recovered[: 8]).view(-1, 1, 28, 28)])
    save_image(comparison, './results_pca/reconstruction_' + str(n_components) + '.png')


if __name__ == '__main__':
    main()
