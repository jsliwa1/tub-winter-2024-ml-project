import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms

from src.modules.config import MNISTBagsConfig
from src.modules.plotting import visualize_gtbags

class MNISTBags(data_utils.Dataset):
    """
    A PyTorch Dataset for generating and loading MNIST images as bags for use in multiple-instance learning (MIL) settings.

    Attributes:
        random (numpy.random.Generator): A random generator initialized with a specific seed.
        positive_num (int): The digit that is considered the positive label in bags (e.g., 9).
        mean_bag_size (int): The average number of instances (images) per bag.
        var_bag_size (float): The variance in the number of instances per bag.
        num_bags (int): The total number of bags in the dataset.
        train (bool): Flag indicating if this dataset instance is for training or testing.
        num_train_samples (int): Number of samples in the training set.
        num_test_samples (int): Number of samples in the test set.
        train_bags_list (list): List of bags for training if `train=True`, each containing a list of images.
        train_labels_list (list): List of labels for each training bag, indicating positive or negative status.
        test_bags_list (list): List of bags for testing if `train=False`, each containing a list of images.
        test_labels_list (list): List of labels for each test bag, indicating positive or negative status.
    """
    def __init__(
        self,
        seed: int,
        positive_num: int,
        mean_bag_size: int,
        var_bag_size: int,
        num_bags: int,
        train: bool,
        test_attention: bool
    ):
        """
        Initializes MnistBags dataset by generating bags of MNIST images.

        Args:
            seed (int): Seed for random number generation.
            positive_num (int): The digit that is considered the positive label in bags (e.g., 9).
            mean_bag_size (int): The average number of instances (images) per bag.
            var_bag_size (float): The variance in the number of instances per bag.
            num_bags (int): The total number of bags in the dataset.
            train (bool): Flag indicating if this dataset instance is for training or testing.
            test_attention (bool): Flag indicating if this dataset is for testing attention mechanisms.
        """
        self.random = np.random.default_rng(seed)

        self.positive_num = positive_num
        self.mean_bag_size = mean_bag_size
        self.var_bag_size = var_bag_size
        self.num_bags = num_bags
        self.train = train
        self.test_attention = test_attention
        self.num_train_samples = 60000
        self.num_test_samples = 10000

        if self.train:
            self.train_bags_list, self.train_labels_list = self._create_bags()
        else:
            self.test_bags_list, self.test_labels_list = self._create_bags()

    def _create_bags(self):
        """
        Creates bags of images and labels from the MNIST dataset.

        Returns:
            tuple: (bags_list, labels_list)
                - bags_list (list of Tensors): Each bag is a list of images.
                - labels_list (list of bools): Each bag label is True if it contains the target digit.
        """
        try:
            loader = data_utils.DataLoader(
                datasets.MNIST('data', train=self.train, download=True, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])),
                batch_size=self.num_train_samples if self.train else self.num_test_samples,
                shuffle=False
            )
        except Exception as e:
            print(f"Error loading data: {e}")
            return [], []
        
        counter = 0
        last_bag = 0
        bags_list = []
        labels_list = []

        for batch_data, batch_labels in loader:
            all_imgs = batch_data
            all_labels = batch_labels

        num_samples = self.num_train_samples if self.train else self.num_test_samples

        while counter < self.num_bags:
            bag_length = max(1, int(self.random.normal(self.mean_bag_size, self.var_bag_size, 1)[0]))

            if self.test_attention:
                indices = []
                positive_indices = np.where(all_labels == self.positive_num)[0]
                
                if len(positive_indices) < 4:
                    print("Insufficient positive samples in dataset for attention testing.")
                    return [], []
                
                selected_positive_indices = self.random.choice(positive_indices, 4, replace=False)
                indices.extend(selected_positive_indices)

                if len(indices) < bag_length:
                    remaining_length = bag_length - len(indices)
                    other_indices = [idx for idx in range(num_samples) if all_labels[idx] != self.positive_num]
                    selected_other_indices = self.random.choice(other_indices, remaining_length, replace=False)
                    indices.extend(selected_other_indices)

                self.random.shuffle(indices)
                indices = torch.LongTensor(indices)
                labels_in_bag = all_labels[indices]
                labels_list.append((labels_in_bag == self.positive_num))
                bags_list.append(all_imgs[indices])

                counter += 1

            else:
                if last_bag == 0:
                    # Positive bag (contains `self.positive_num`)
                    indices = torch.LongTensor(self.random.integers(0, num_samples, bag_length))
                    labels_in_bag = all_labels[indices]

                    if self.positive_num in labels_in_bag:
                        labels_list.append(labels_in_bag == self.positive_num)
                        bags_list.append(all_imgs[indices])
                        last_bag = 1
                        counter += 1
                else:
                    # Negative bag (excludes `self.positive_num`)
                    indices = []
                    while len(indices) < bag_length:
                        idx = self.random.integers(0, num_samples)
                        if all_labels[idx] != self.positive_num:
                            indices.append(idx)

                    indices = torch.LongTensor(indices)
                    labels_in_bag = all_labels[indices]
                    labels_list.append(labels_in_bag == self.positive_num)
                    bags_list.append(all_imgs[indices])
                    last_bag = 0
                    counter += 1

        return bags_list, labels_list


    def __len__(self):
        """
        Returns the number of bags in the dataset.

        Returns:
            int: Number of bags.
        """
        return len(self.train_labels_list) if self.train else len(self.test_labels_list)

    def __getitem__(self, index):
        """
        Retrieves a bag and its label.

        Args:
            index (int): Index of the bag.

        Returns:
            tuple: (bag, label)
                - bag (Tensor): Images in the bag.
                - label (list): [True if bag contains target, individual labels in bag].
        """
        bag = self.train_bags_list[index] if self.train else self.test_bags_list[index]
        label = [max(self.train_labels_list[index]), self.train_labels_list[index]] if self.train else [max(self.test_labels_list[index]), self.test_labels_list[index]]

        return bag, label


def test_MnistBags(
    train_loader: data_utils.DataLoader,
    test_loader: data_utils.DataLoader
):
    """
    Evaluates the MNIST bags dataset by summarizing bag statistics for training and test sets.

    Calculates and prints the number of positive bags (those containing the target number) and
    the distribution of instances per bag in both training and test datasets.

    Args:
        train_loader (data_utils.DataLoader): DataLoader for training bags with MNIST instances.
        test_loader (data_utils.DataLoader): DataLoader for test bags with MNIST instances.

    Prints:
        - Number of positive bags in the training set vs. total training bags.
        - Number of positive bags in the test set vs. total test bags.
        - Mean, maximum, and minimum number of instances per bag in both the training and test sets.
    """
    train_bag_size_lst = []
    count_positives_tr = 0
    for batch_idx, (bag, label) in enumerate(train_loader):
        train_bag_size_lst.append(int(bag.squeeze(0).size()[0]))
        count_positives_tr += label[0].numpy()[0]
    print('Number positive train bags: {}/{}\n'
          'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
        count_positives_tr, len(train_loader),
        np.mean(train_bag_size_lst), np.max(train_bag_size_lst), np.min(train_bag_size_lst)))

    len_bag_list_test = []
    count_positives_te = 0
    for batch_idx, (bag, label) in enumerate(test_loader):
        len_bag_list_test.append(int(bag.squeeze(0).size()[0]))
        count_positives_te += label[0].numpy()[0]
    print('Number positive test bags: {}/{}\n'
          'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
        count_positives_te, len(test_loader),
        np.mean(len_bag_list_test), np.max(len_bag_list_test), np.min(len_bag_list_test)))

def test_visualization(
    train_loader: data_utils.DataLoader,
    positive_num: int,
    show: bool = False
):
    """
    Tests the visualization of bags with individual labels.

    Args:
        train_loader (data_utils.DataLoader): DataLoader for training bags with MNIST instances.
        positive_num (int): The digit that is considered the positive label in bags (e.g., 9).
        show (bool): If True, displays the plot; otherwise, saves it to file.
    """
    for batch_idx, (bag, label) in enumerate(train_loader):
        print(f'Train Batch {batch_idx + 1} - Bag size: {bag.shape} - Bag Label: {label[0].numpy()}')
        visualize_gtbags(bag, label, batch_idx, positive_num, show, "./logs/misc/data")
    

if __name__ == "__main__":
    train_config = MNISTBagsConfig(
        seed=1,
        positive_num=9,
        mean_bag_size=10,
        var_bag_size=2,
        num_bags=5,
        train=True,
        test_attention=True
    )
    test_config = MNISTBagsConfig(
        seed=1,
        positive_num=9,
        mean_bag_size=10,
        var_bag_size=2,
        num_bags=5,
        train=False,
        test_attention=False
    )
    train_loader = data_utils.DataLoader(MNISTBags(**train_config.__dict__),
                                         batch_size=1,
                                         shuffle=True)
    test_loader = data_utils.DataLoader(MNISTBags(**test_config.__dict__),
                                        batch_size=1,
                                        shuffle=False)
    
    test_MnistBags(train_loader=train_loader, test_loader=test_loader)
    test_visualization(train_loader=train_loader, test_loader=test_loader, positive_num=train_config.positive_num, show=False)
    print("Dataset test passed!")