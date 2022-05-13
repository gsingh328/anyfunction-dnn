import os
import sys
import argparse
import logging
from time import localtime, strftime, perf_counter
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import MultiStepLR, StepLR

import numpy as np


def train(args, model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(args, model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            loss = criterion(output, target)

            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            if batch_idx % args.log_interval == 0:
                logging.info('Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch_idx * len(data), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader), loss.item()))

    test_loss /= len(test_loader.dataset)

    logging.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, correct / len(test_loader.dataset)


def make_dir(args):
    root = args.output_dir

    time = strftime("%m-%d-%H-%M-%S", localtime())
    dataset = args.dataset
    log_name = '%s.log' % time
    ckpt_name = '%s_ckpt' % time

    log_dir = os.path.join(root, args.dataset)
    log_dir = os.path.join(log_dir, args.quant_mode)

    log_file = os.path.join(log_dir, log_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    ckpt_dir = os.path.join(log_dir, ckpt_name)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    return log_file, ckpt_dir


def handle_checkpointing(model, optimizer, scheduler, epoch, best_val_loss,
        best_val_score, ckpt_dir, is_best):
    def save_model(ckpt_path):
        logging.info(f'Saving checkpoint: {ckpt_path}')
        torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'best_val_score': best_val_score
                }, ckpt_path)

    ckpt_path = os.path.join(ckpt_dir, f'checkpoint_last.pt')
    save_model(ckpt_path)
    if is_best:
        logging.info(f'Found new best model.')
        ckpt_path = os.path.join(ckpt_dir, f'checkpoint_best.pt')
        save_model(ckpt_path)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--dataset', type=str, default='cifar10', metavar='DS',
                        choices=['mnist', 'cifar10',],
                        help='Dataset (default: cifar10)')
    parser.add_argument('--model', type=str, default='MobileNet', metavar='MD',
                        help='Model (default: MobileNet)')
    parser.add_argument('--quant-mode', type=str, default='none',
                        choices=['none', 'symmetric',],
                        help='Quantization mode (default: none).')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 100)')
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
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Folder name to store logs and checkpoints (default: outputs')
    parser.add_argument('--dataset-dir', type=str, default='~/icml_data/',
                        help='Folder where datasets are stored (default: "~/icml_data/")')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint to resume from (default: None)')
    parser.add_argument('--no-save', action='store_true',
                        help='Don\'t save checkpoints.')
    parser.add_argument('--finetune', action='store_true',
                        help='Resets epoch and validation score' +\
                        ' that is loaded from pre-trained checkpoint.')
    args = parser.parse_args()

    log_file, ckpt_dir = make_dir(args)
    logging.basicConfig(
        level=logging.DEBUG,
        format='-- %(asctime)s -- %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S%p',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info(f'Arguments:\n{args}')

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    logging.info(f'Using device: {device}')

    torch.manual_seed(args.seed)

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 4,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    if args.dataset == "mnist":
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        trainset = datasets.MNIST('~/icml_data', train=True, download=False,
            transform=transform)
        testset = datasets.MNIST('~/icml_data', train=False, download=False,
            transform=transform)

        from anyfunction.models import mnist as dataset_models

    elif args.dataset == "cifar10":
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

        trainset = datasets.CIFAR10('~/icml_data', train=True, download=False,
            transform=transform_train)
        testset = datasets.CIFAR10( root='~/icml_data', train=False, download=False,
            transform=transform_test)

        from anyfunction.models import cifar10 as dataset_models

    train_loader = torch.utils.data.DataLoader(trainset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(testset, **test_kwargs)

    if not hasattr(dataset_models, args.model):
        logging.critical(f'Could not find model "{args.model}"')
        exit(1)
    Net = getattr(dataset_models, args.model)
    model = Net(quant_mode=args.quant_mode).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()
    scheduler = MultiStepLR(optimizer, milestones=[int(args.epochs * 0.8)], gamma=0.1)
    start_epoch = 1
    best_val_loss = np.inf
    best_val_score = -np.inf

    if args.resume is not None:
        if not os.path.isfile(args.resume):
            logging.critical(f'Invalid checkpoint file: {args.resume}')
            exit(1)
        logging.info(f'Resuming from: {args.resume}')
        checkpoint = torch.load(args.resume)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if not args.finetune:
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']

    logging.info('=' * 100)
    logging.info(f'Model:\n{model}')
    logging.info('=' * 100)
    logging.info(f'Optimizer:\n{optimizer}')
    logging.info('=' * 100)
    logging.info(f'Scheduler:\n{scheduler}')
    logging.info('=' * 100)
    logging.info(f'\nStart Epoch: {start_epoch}')
    logging.info(f'Best Validation Loss: {best_val_loss}')
    logging.info('\nStarting training...')

    for epoch in range(start_epoch, args.epochs + 1):
        st_time = perf_counter()
        train(args, model, device, train_loader, criterion, optimizer, epoch)
        val_loss, val_score = test(args, model, device, test_loader, criterion)
        scheduler.step()
        end_time = perf_counter()

        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        best_val_score = max(val_score, best_val_score)
        handle_checkpointing(model, optimizer, scheduler, epoch,
            best_val_loss, best_val_score, ckpt_dir, is_best)

        time_delta = end_time - st_time
        td = timedelta(seconds=time_delta * (args.epochs + 1 - epoch))
        days, hrs, mins = td.days, td.seconds // 3600, td.seconds // 60 % 60
        time_when_complete = (datetime.today() + td).strftime('%I:%M:%S%p')

        logging.info('-' * 100)
        logging.info(f'Best Validation Score at Epoch [{epoch}] ==> {best_val_score:.4f}')
        logging.info('')
        logging.info(f'Time for Epoch [{epoch}] ==> {time_delta:.4f} secs')
        logging.info(f'Time to complete training [DAYS:HRS:MIN:SEC] ==> {days:02}:{hrs:02}:{mins:02}')
        logging.info(f'Time when complete ==> {time_when_complete}')
        logging.info('-' * 100)

if __name__ == '__main__':
    main()
