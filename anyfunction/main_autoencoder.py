import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


DATASET="MNIST"

if DATASET == "MNIST":
    from mnist_model import BasicMLP, BasicMLP_AnyF
    from autoencoder import MNISTAutoEncoder as AutoEncoder
elif DATASET == "CIFAR10":
    from cifar10_model import BasicMLP, BasicMLP_AnyF
    from autoencoder import CIFAR10AutoEncoder as AutoEncoder


def train(args, model, device, train_loader, optimizer, criterion, epoch, autoencoder=None, training_autoencoder=False):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        if autoencoder is not None:
            embeddings = autoencoder.encoder(data).squeeze()
            output = model(embeddings)
        else:
            output = model(data)

        if training_autoencoder:
            loss = criterion(output, data)
        else:
            loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader, criterion, autoencoder=None, training_autoencoder=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            if autoencoder is not None:
                embeddings = autoencoder.encoder(data).squeeze()
                output = model(embeddings)
            else:
                output = model(data)

            if training_autoencoder:
                loss = criterion(output, data)
            else:
                loss = criterion(output, target)

            test_loss += loss.item()

            if not training_autoencoder:
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    if training_autoencoder:
        print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    else:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



def get_autoencoder(args, device, train_loader, test_loader):
    model = AutoEncoder().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.7)

    criterion = nn.MSELoss()

    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, 5+1):
        train(args, model, device, train_loader, optimizer, criterion, epoch, training_autoencoder=True)
        test(model, device, test_loader, criterion, training_autoencoder=True)
        # scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
    return model


def get_encoded_mlp(args, device, Net, autoencoder, train_loader, test_loader):
    model = Net().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.7)

    criterion = nn.CrossEntropyLoss()

    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, criterion, epoch, autoencoder=autoencoder)
        test(model, device, test_loader, criterion, autoencoder=autoencoder)
        # scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
    return model


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--wd', type=float, default=1e-5, metavar='WD',
                        help='weight decay (default: 1e-5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 4,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    if DATASET == "MNIST":
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        trainset = datasets.MNIST('~/icml_data/', train=True, download=False,
                        transform=transform)
        testset = datasets.MNIST('~/icml_data/', train=False, download=False,
                        transform=transform)

    elif DATASET == "CIFAR10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = datasets.CIFAR10('~/icml_data', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10( root='~/icml_data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(trainset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(testset, **test_kwargs)

    print("=" * 50)
    print("AUTOENCODER")
    print("=" * 50)
    autoencoder = get_autoencoder(args, device, train_loader, test_loader)

    # print("=" * 50)
    # print("BASIC MLP")
    # print("=" * 50)
    # basic_mlp = get_encoded_mlp(args, device, BasicMLP, autoencoder, train_loader, test_loader)

    print("=" * 50)
    print("ANYF MLP")
    print("=" * 50)
    anyf_mlp = get_encoded_mlp(args, device, BasicMLP_AnyF, autoencoder, train_loader, test_loader)

    anyf_mlp.af1.debug_print_graph()


if __name__ == '__main__':
    main()
