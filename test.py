import unittest
import os

from data.dataset import CustomDataset


class TestCustomDataset(unittest.TestCase):
    MNIST_DATA = r"./data/mnist"
    CIFAR10 = r"./data/cifar10"
    
    def test_get_files(self):
        ds = CustomDataset("test_dataset", self.MNIST_DATA, train_mode=True)
        imgs_num = 60000
        self.assertEqual(imgs_num, len(ds.get_images(ds.ex_dir)), "60000 images in the 'train' folder")
        self.assertEqual(0, len(ds.get_images(ds.l_dir)), "0 images in the 'label' folder")

    def test_get_labels(self):
        ds = CustomDataset("test_dataset", self.MNIST_DATA, train_mode=True)
        lab_num = 60000
        self.assertEqual(0, len(ds.get_labels(ds.ex_dir)), "0 labels in the 'train' folder")
        self.assertEqual(lab_num, len(ds.get_labels(ds.l_dir)), "60000 labels in the 'labels' folder")

    def test_join_data(self):
        ds = CustomDataset("test_dataset", self.MNIST_DATA, train_mode=True)
        tuples_num = 60000
        self.assertEqual(tuples_num, len(ds.data), "60000 tuples in the 'train' folder")

    def test_dataset_len(self):
        ds = CustomDataset("test_dataset", self.MNIST_DATA, train_mode=True)
        tuples_num = 60000
        self.assertEqual(tuples_num, len(ds), "60000 tuples in the dataset")

    def test_get_item(self):
        ds = CustomDataset("test_dataset", self.MNIST_DATA, train_mode=True)
        t1 = ds.__getitem__(0)
        self.assertEqual((1,28,28), t1[0].size())
        self.assertEqual("5", t1[1])

        t2 = ds.__getitem__(1)
        self.assertEqual((1,28,28), t2[0].size())
        self.assertEqual("0", t2[1])


if __name__ == "__main__":
    unittest.main()
