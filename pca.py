import numpy as np
import torch

from sklearn.decomposition import PCA
from torchvision import datasets, transforms
from torchvision.utils import save_image


n_components = 2


def main():
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), shuffle=True)
    test_set = datasets.MNIST('./data', train=False, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=True)
    for data, targets in train_loader:
        train_data = data.view(-1, 784).numpy()
        train_labels = targets.numpy()
    for data, targets in test_loader:
        test_data = data.view(-1, 784).numpy()
        test_labels = targets.numpy()
    pca = PCA(n_components=n_components)
    pca.fit(train_data)
    test_reduced = pca.transform(test_data)
    test_recovered = pca.inverse_transform(test_reduced)
    comparison = torch.cat([torch.from_numpy(test_data[:8]).view(-1, 1, 28, 28),
                            torch.from_numpy(test_recovered[: 8]).view(-1, 1, 28, 28)])
    save_image(comparison, './results_pca/reconstruction_' + str(n_components) + '.png')
    sample = pca.inverse_transform(np.random.randn(64, n_components))
    save_image(torch.from_numpy(sample).view(-1, 1, 28, 28), './results_pca/sample_' + str(n_components) + '.png')
    print(-pca.score(test_data))


if __name__ == '__main__':
    main()
