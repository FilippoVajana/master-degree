from engine.tester import Tester
import train_model as trmod
import engine
import torchvision.transforms as transforms
import torch
import sys
import os
import logging as log
import datetime
import argparse
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [15, 5]


log.basicConfig(level=log.DEBUG,
                format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')

DATA_DICT = {
    'mnist': './data/mnist',
    'no-mnist': './data/no-mnist'
}
DEVICE = 'cpu'
BATCH_SIZE = 128
MAX_ITEMS = 1500


def get_device():
    DEVICE = trmod.get_device()
    return DEVICE


def plot_sample(tensors):
    plt.tight_layout()
    fig = plt.figure(figsize=(15, 5), dpi=200)
    lim = (0, 27)
    ticks = [0, 14, 27]
    labels = [0, 14, 28]
    axs = fig.subplots(1, 3)
    for idx, ax in enumerate(axs):
        t_img = tensors[idx][0].squeeze()
        ax.imshow(t_img, cmap='gray')
        ax.set_xlabel(tensors[idx][1].item())
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    plt.show()


def load_model_state(model_cls: str, model_path: str, device: str):
    try:
        # get model instance
        model_obj = getattr(engine, model_cls)()
        log.debug(f"Initialized model: {model_obj.__class__.__name__}")

        # load model state dict
        model_obj.load_state_dict(torch.load(model_path, map_location=device))
        log.info(f"Loaded model state: {model_path}")
        return model_obj

    except Exception as exc:
        log.error(f"Error loading model state dict: {exc}")
        sys.exit(1)


def save_results(path: str, results: dict):
    '''Saves the results dict (filename.csv, dataframe).
    '''
    for key in results.keys():
        p = os.path.join(path, key)
        results[key].to_csv(p, index=True)
        log.info(f"Saved test result: {p}")


def do_test(model_cls: str, state_dict_path: str, device: str, directory: str, max_items=-1):
    '''Loads a trained model and perform a list of tests.
    The method saves results as .csv files and returns the results dictionary.
    '''
    log.info("Started testing phase.")

    # check device
    if device is None:
        device = get_device()

    # load model
    model_obj = load_model_state(model_cls, state_dict_path, device)

    # in distribution mnist
    t_res_dict = dict()
    log.info("Testing MNIST")
    t_res_dict["mnist.csv"] = test_regular_data(model_obj, max_items=max_items)

    # out-of-distribution notmnist
    log.info("Testing Out-Of-Distribution")
    t_res_dict["nomnist.csv"] = test_ood_data(model_obj, max_items=max_items)

    # rotation skew
    rotation_range = range(15, 180 + 15, 15)
    for rotation_value in rotation_range:
        log.info(f"Testing Rotated {rotation_value} MNIST")
        df = test_rotated_data(model_obj, "mnist", rotation_value, max_items=max_items)
        t_res_dict[f"mnist_rotate{rotation_value}.csv"] = df

    # pixel shift skew
    shift_range = range(2, 14 + 2, 2)
    img_size = 28
    for shift_value in shift_range:
        shift_value /= img_size
        log.info(f"Testing Shifted {int(shift_value * img_size)}px MNIST")
        df = test_shifted_data(model_obj, "mnist", shift_value, max_items=max_items)
        t_res_dict[f"mnist_shift{int(shift_value * img_size)}.csv"] = df

    # save test results
    if directory is not None:
        save_results(path=directory, results=t_res_dict)
    return t_res_dict


def test_regular_data(model, dataset_name="mnist", max_items=-1):
    # get dataloader
    dataloader = engine.ImageDataLoader(
        data_folder=DATA_DICT[dataset_name],
        batch_size=BATCH_SIZE,
        shuffle=False,
        transformation=None
    ).build(train_mode=False, max_items=max_items, validation_ratio=0)

    # dl_iter = iter(dataloader[0])
    # samples = [
    #     next(dl_iter),
    #     next(dl_iter),
    #     next(dl_iter)
    # ]
    # plot_sample(samples)

    # test model
    tester = Tester(model, device=DEVICE, is_ood=False)
    df = tester.test(dataloader[0])
    return df


def test_ood_data(model, dataset_name="no-mnist", max_items=-1):
    # get dataloader
    dataloader = engine.ImageDataLoader(
        data_folder=DATA_DICT[dataset_name],
        batch_size=BATCH_SIZE,
        shuffle=False,
        transformation=None
    ).build(train_mode=False, max_items=max_items, validation_ratio=0)

    # dl_iter = iter(dataloader[0])
    # samples = [
    #     next(dl_iter),
    #     next(dl_iter),
    #     next(dl_iter)
    # ]
    # plot_sample(samples)

    # test model
    tester = Tester(model, device=DEVICE, is_ood=True)
    df = tester.test(dataloader[0])
    return df


def test_rotated_data(model, dataset_name="mnist", rotation_value=45, max_items=-1):
    transformation = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation((rotation_value, rotation_value)),
        transforms.ToTensor()
    ])

    # get dataloader
    dataloader = engine.ImageDataLoader(
        data_folder=DATA_DICT[dataset_name],
        batch_size=BATCH_SIZE,
        shuffle=False,
        transformation=transformation
    ).build(train_mode=False, max_items=max_items, validation_ratio=.0)

    # if rotation_value == 45:
    #     dl_iter = iter(dataloader[0])
    #     samples = [
    #         next(dl_iter),
    #         next(dl_iter),
    #         next(dl_iter)
    #     ]
    #     plot_sample(samples)
    # else:
    #     return None

    # test model
    tester = Tester(model, device=DEVICE, is_ood=False)
    df = tester.test(dataloader[0])
    return df


def test_shifted_data(model, dataset_name="mnist", shift_value=.5, max_items=-1):
    transformation = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine(0, translate=(shift_value, shift_value)),
        transforms.ToTensor()
    ])

    # get dataloader
    dataloader = engine.ImageDataLoader(
        data_folder=DATA_DICT[dataset_name],
        batch_size=BATCH_SIZE,
        shuffle=False,
        transformation=transformation
    ).build(train_mode=False, max_items=max_items, validation_ratio=.0)

    # if shift_value == (10 / 28):
    #     dl_iter = iter(dataloader[0])
    #     samples = [
    #         next(dl_iter),
    #         next(dl_iter),
    #         next(dl_iter)
    #     ]
    #     plot_sample(samples)
    # else:
    #     return None

    # test model
    tester = Tester(model, device=DEVICE, is_ood=False)
    df = tester.test(dataloader[0])
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test DNN models.")
    parser.add_argument('-ms', '--mstate', type=str,
                        action='store', help='Model state dict path.')
    parser.add_argument('-mn', '--mname', type=str,
                        action='store', help='Model class name.')
    parser.add_argument('-o', '--outdir', type=str,
                        action='store', help='Test results output directory.')
    parser.add_argument('--batch-size', type=int,
                        action='store', help='Test batch size.')
    parser.add_argument('-short', action='store_true', default=False, help='Use less input data.')

    args = parser.parse_args()

    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size

    ###############
    # TEST MODEL #
    if args.short == True:
        max_items = MAX_ITEMS
    else:
        max_items = -1

    results_df_dict = do_test(model_cls=args.mname, state_dict_path=args.mstate,
                              device=None, directory=args.outdir, max_items=max_items)
