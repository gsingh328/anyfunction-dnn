# Any-Function for DNNs

MNIST and CIFAR10 need to be installed in `~/icml_data/`

To run the MNIST or CIFAR10, set the dataset parameter approriately in line 10 in `main.py`.

Currently, the Any-Function-1 layer needs to be commented in/out if you want to run the tests with our without that layer. It currently only supports 2D tensors for feed-forward networks. Future port will be made to support n-D tensors.
