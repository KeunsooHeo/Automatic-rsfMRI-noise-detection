import torch
import torch.nn as nn
from torchsummary import summary
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import cbam
import scipy.io
from config import config
from util import Hdf5
from model import Resnet_sm, Resnet_ts

parser = config().parse_args()


def load_features():
    filename_format = "./test/tensor_{}_{}_{}.mat"
    features = {}
    for dim in [3]:
        for i in [1]:
            for j in ["before", "channel", "spatial", "after"]:
                if dim == 1 and j == "spatial":
                    continue
                features["{}_{}_{}".format(dim,i,j)] = scipy.io.loadmat(filename_format.format(dim,i,j))["weight"]
    return features


def file_visualization(file, data, channel):
    before = torch.load(file.replace(".pt","_before.pt"))
    after = torch.load(file.replace(".pt","_after.pt"))
    before = before.cpu().detach().numpy()
    after = after.cpu().detach().numpy()

    background = data[0, 0, :, :, data.shape[4]//2].cpu().detach().numpy()
    background = np.abs(background)
    background = 255 * (background - background.min()) / (background.max() - background.min())

    s = before.shape[4]//2
    before = before[0, 0, :, :, s]
    print("before : ",before.min(),before.max() )
    before = np.abs(before)
    before = 255 * abs(before) / (before.max() - before.min())
    before = np.array(before, dtype=np.uint8)
    print("before normal : ", before.min(), before.max())


    after = after[0, 0, :, :, s]
    print("after : ", after.min(), after.max())
    after = np.abs(after)
    after = 255 * abs(after) / (after.max() - after.min())
    after = np.array(after, dtype=np.uint8)
    print("after normal : ", after.min(), after.max())

    plt.subplot(1, 3, 1)
    plt.title("background")
    plt.imshow(background, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 3, 2)
    plt.title(file.replace(".pt","_before.pt"))
    plt.imshow(before, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 3, 3)
    plt.title(file.replace(".pt","_after.pt"))
    plt.imshow(after, cmap='gray', vmin=0, vmax=255)
    plt.show()



def main():
    # save_feature()
    features = load_features()

    max_index = {}
    set = {}
    for p, inplane in enumerate([1]):
        a = features["3_{}_channel".format(inplane)].squeeze()
        i, max = np.argmax(a), len(a)
        max_index["{}".format(inplane)] = i
        print(i, max)
        print(a.shape)

        # channel attention map
        plt.suptitle("channel attention map")
        plt.subplot(2, 2, p+1)
        plt.title("layer {}".format(p+1))
        plt.bar(range(len(a)),a)
        plt.show()

        # ca max channel
        _before = features["3_{}_before".format(inplane)][0, i]
        _after = features["3_{}_after".format(inplane)][0, i]

        print(_before.shape, _after.shape)

        # save before and after
        set["layer{}_A".format(p + 1)] = _before
        set["layer{}_B".format(p + 1)] = _after

        before = _before[:, :, _before.shape[-1] // 2]
        before = np.array(abs_norm(before), dtype=np.uint8)
        after = _after[:, :, _after.shape[-1] // 2]
        after = np.array(abs_norm(after), dtype=np.uint8)

        plt.subplot(2, 3, 1)
        plt.title("before")
        plt.imshow(before, cmap='gray', vmin=0, vmax=255)
        plt.subplot(2, 3, 4)
        plt.title("after")
        plt.imshow(after, cmap='gray', vmin=0, vmax=255)

        before = _before[:, _before.shape[-2] // 2, :]
        before = np.array(abs_norm(before), dtype=np.uint8)
        after = _after[:, _after.shape[-2] // 2, :]
        after = np.array(abs_norm(after), dtype=np.uint8)

        plt.subplot(2, 3, 2)
        plt.title("before")
        plt.imshow(before, cmap='gray', vmin=0, vmax=255)
        plt.subplot(2, 3, 5)
        plt.title("after")
        plt.imshow(after, cmap='gray', vmin=0, vmax=255)

        before = _before[_before.shape[-3] // 2, :, :]
        before = np.array(abs_norm(before), dtype=np.uint8)
        after = _after[_after.shape[-3] // 2, :, :]
        after = np.array(abs_norm(after), dtype=np.uint8)

        plt.subplot(2, 3, 3)
        plt.title("before")
        plt.imshow(before, cmap='gray', vmin=0, vmax=255)
        plt.subplot(2, 3, 6)
        plt.title("after")
        plt.imshow(after, cmap='gray', vmin=0, vmax=255)

        plt.show()

    hdf5 = Hdf5(dataset=parser.dataset, is_train=True, num_source=parser.num_source, data_keys=parser.data_keys,
                source_dir=parser.path_source)
    data = hdf5.getDataDicByIndex(0)['data'].squeeze()
    print(data.shape)
    set["data"] = data

    scipy.io.savemat("test/attention_sample.mat", set)

    plt.show()

def abs_norm(img):
    img = abs(img)
    return 255 * (img) / (img.max() - img.min())

if __name__ == "__main__":
    main()