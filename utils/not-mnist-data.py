import os
import tarfile
import imageio
from PIL import Image
import tqdm
import numpy as np

def get_nomnist():
    classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    tar_path = "./data/no-mnist/archive/notMNIST_small.tar"
    tmp_path = "./data/no-mnist/archive/tmp"

    img_arr = []
    lab_arr = []

    with tarfile.open(tar_path) as tar:
        tar_root = tar.next().name
        for c in tqdm.tqdm(classes):
            tqdm.tqdm.write(f"Extracting class {c}")
            # get class members
            files = [f for f in tar.getmembers() if f.name.startswith(f"{tar_root}/{c}")]

            # extract members
            if not os.path.exists(tmp_path):
                os.mkdir(tmp_path)

            for f in files:
                # read images
                try:
                    f_obj = tar.extractfile(f)
                    img = imageio.imread(f_obj)                    
                    arr = np.asarray(img)
                    arr = np.expand_dims(arr, axis=0)
                    img_arr.append(arr)
                    lab_arr.append(ord(c))
                except Exception:
                    continue

            tqdm.tqdm.write(str(len(files)))

    os.rmdir(tmp_path)
    return img_arr, lab_arr


def save_data(arr: np.ndarray, name: str, path: str):
    size = arr.size * arr.itemsize / 1e6
    tqdm.tqdm.write(f"Saving {name} ndarray [{size} MB]")
    np.save(os.path.join(path, name), arr)

if __name__ == "__main__":
    # make dirs
    DATA_ROOT = "./data/no-mnist"
    TEST_DIR = os.path.join(DATA_ROOT, "test")
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)

    # get data
    img_list, lab_list = get_nomnist()

    # save data
    img_arr = np.asarray(img_list)
    save_data(img_arr, "images", TEST_DIR)
    lab_arr = np.asarray(lab_list)
    save_data(lab_arr, "labels", TEST_DIR)
