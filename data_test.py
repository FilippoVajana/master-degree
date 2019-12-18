import unittest

import engine
from engine.dataset import CustomDataset
from engine.dataloader import *


class TestCustomDataset(unittest.TestCase):
    MNIST_DATA = r"./data/mnist/train"
    CIFAR10 = r"./data/cifar10"

    def test_dataset_len(self):
        ds = CustomDataset(self.MNIST_DATA)
        tuples_num = 60000
        self.assertEqual(tuples_num, len(ds.data), "60000 tuples in the dataset")

    def test_get_item(self):
        ds = CustomDataset(self.MNIST_DATA)
        t1 = ds.__getitem__(0)
        self.assertEqual((1, 28, 28), t1[0].size())
        self.assertEqual("5", t1[1])

        t2 = ds.__getitem__(1)
        self.assertEqual((1, 28, 28), t2[0].size())
        self.assertEqual("0", t2[1])


# class ImageDataLoaderTest(unittest.TestCase):
#     MNIST_DATA = r"./data/mnist"

#     def test_dataloader_build(self):       
#         dataloader = ImageDataLoader(
#             data_folder=self.MNIST_DATA,
#             batch_size=1,
#             shuffle=False,
#             train_mode=True).dataloader      
#         count = 0
#         for _ in iter(dataloader):
#             count += 1
#         self.assertEqual(count, len(dataloader.dataset))



if __name__ == "__main__":
    unittest.main()
