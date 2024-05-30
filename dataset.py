import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

DATA_ROOT = "/home/hiroki/dev/datasets"

def load_data(batch_size):
    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    train_data = datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=False, transform=train_transforms
    )

    val_data = datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=False, transform=val_transforms
    )

    test_data = datasets.CIFAR10(
        root=DATA_ROOT, train=False, download=False, transform=test_transforms
    )

    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=True)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, val_loader, test_loader, classes
