# Image-Classification using Convolutional Neural Network

In this project, we'll classify images from the CIFAR-10 dataset (https://www.cs.toronto.edu/~kriz/cifar.html). The dataset consists of airplanes, dogs, cats, and other objects. We'll preprocess the images, then train a convolutional neural network on all the samples.

Download the CIFAR-10 dataset for python (https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz).

### Data

CIFAR-10 is an established computer-vision dataset used for object recognition. It is a subset of the 80 million tiny images dataset and consists of 60,000 32x32 color images containing one of 10 object classes, with 6000 images per class. It was collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.

### Get the data by running the following function

    from urllib.request import urlretrieve
    from os.path import isfile, isdir
    from tqdm import tqdm
    import tarfile
    
    cifar10_dataset_folder_path = 'cifar-10-batches-py'
    
    class DLProgress(tqdm):
        last_block = 0
        
        def hook(self, block_num=1, block_size=1, total_size=None):
            self.total = total_size
            self.update((block_num - self.last_block) * block_size)
            self.last_block = block_num
          
    if not isfile('cifar-10-python.tar.gz'):
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset       
            urlretrieve('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz','cifar-10-python.tar.gz',pbar.hook)
    
    if not isdir(cifar10_dataset_folder_path):
        with tarfile.open('cifar-10-python.tar.gz') as tar:
            tar.extractall()
            tar.close()
            
### Data exploration:

The dataset is broken into batches to prevent pc from running out of memory. The CIFAR-10 dataset consists of 5 batches, named data_batch_1, data_batch_2, etc.. Each batch contains the labels and images that are one of the following:

airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck

### We can use the code in helper.py
