import os
import engine.dataloader as dat

if __name__ == "__main__":
    DATA_ROOT = "./data/no-mnist"
    TEST_DIR = os.path.join(DATA_ROOT)
    dl = dat.ImageDataLoader(TEST_DIR, 1, False, False, -1)
    for img, lab in dl.dataloader:
        print(img.shape)
