import cbam, resnet
from model import Resnet_sm_gradcam, Resnet_ts_gradcam
import os
from util import Hdf5, get_state_dict
import torch.nn as nn
import cv2
import numpy as np
import numpy.ma as ma
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable, Function
from gcam import gcam
import matplotlib.pyplot as plt
import cv2
import nibabel as nib
import pandas as pd
from scipy.io import loadmat
from datetime import datetime
import matplotlib as mpl
from matplotlib.collections import LineCollection
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
from PIL import Image
from copy import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc
from scipy.fft import fftfreq
from matplotlib.patches import Rectangle
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def plot(img):
    a,b,c = [i//2 for i in img.shape]
    plt.figure()
    plt.subplot(2,2,3)
    _img = img[a,:,:]
    #_img = np.uint8(255 * _img)
    plt.imshow(_img.T, cmap='gray')
    plt.subplot(2,2,2)
    _img = img[:,b,:]
    #_img = np.uint8(255 * _img)
    plt.imshow(_img, cmap='gray')
    plt.subplot(2,2,1)
    _img = img[:,:,c]
    #_img = np.uint8(255 * _img)
    plt.imshow(_img, cmap='gray')
    plt.colorbar()
    plt.show()

def show_cam(mask, img):
    assert mask.shape == img.shape, "shape is not same, {} {}".format(mask.shape,img.shape)
    a, b, c = [i // 2 for i in img.shape]
    plt.figure()
    img = abs(img)

    plt.subplot(2, 2, 3)
    _img = img[a, :, :]
    _mask = mask[a, :, :]
    cam = _mask + np.float32(_img)
    #cam = cam / np.max(cam)
    #cam = np.uint8(255 * cam)
    plt.imshow(cam.T, cmap='gray')

    plt.subplot(2, 2, 2)
    _img = img[:, b, :]
    _mask = mask[:, b, :]
    cam = _mask + np.float32(_img)
    #cam = cam / np.max(cam)
    #cam = np.uint8(255 * cam)
    plt.imshow(cam, cmap='gray')

    plt.subplot(2, 2, 1)
    _img = img[:, :, c]
    _mask = mask[:, :, c]
    cam = _mask + np.float32(_img)
    #cam = cam / np.max(cam)
    #cam = np.uint8(255 * cam)
    plt.imshow(cam, cmap='gray')
    #plt.show()


def get_sample(dataset):
    if dataset == "hcp":
        rootdir = '/home/user/mailab_nas/heo/Denoising/data/HCP_hp2000_All_FIX/'
    elif dataset == "std":
        rootdir = '/home/user/mailab_nas/heo/Denoising/data/WhII_Standard_FIX/'
    elif dataset == "mb6":
        rootdir = '/home/user/mailab_nas/heo/Denoising/data/WhII_MB6_FIX/'
    elif dataset == "bcp":
        rootdir = '/home/user/mailab_nas/heo/Denoising/data/BCP_for_FIX/'
    else:
        raise ValueError("Unknown dataset : {}".format(dataset))

    sample_dir = os.path.join(rootdir, "Sample1/filtered_func_data.ica/melodic_IC.nii.gz")
    sample = nib.load(sample_dir)

    return sample

def get_timestamp(dataset, is_cbam):
    t = "_"
    if is_cbam:
        if dataset == "hcp":
            t = t + "20200929150300"#"20200903092921"
        elif dataset == "bcp":
            t = t + "20201020112412"
        elif dataset == "mb6":
            t = t + "20201014151312"#"20200903092923"
        elif dataset == "std":
            t = t + "20201014151322"#"20200903092926"
        else:
            raise ValueError("Unknown dataset : {}".format(dataset))
    else:
        if dataset == "hcp":
            t = t + "20200907110543"
        elif dataset == "bcp":
            t = t + "20200907110637"
        elif dataset == "mb6":
            t = t + "20200914182941"
        elif dataset == "std":
            t = t + "20200914183005"
        else:
            raise ValueError("Unknown dataset : {}".format(dataset))

    return t

def get_prefix(dataset):
    prefix = os.path.join(".", "data")
    if dataset == "hcp":
        prefix = os.path.join(prefix, "HCP_hp2000_All_MNI")
    elif dataset == "bcp":
        prefix = os.path.join(prefix, "BCP_p1_1_hdf5")
    elif dataset == "mb6":
        prefix = os.path.join(prefix, "WhII_MB6_MNI")
    elif dataset == "std":
        prefix = os.path.join(prefix, "WhII_Standard_MNI")
    return prefix


def get_name_sample_comp(dataset, sample, comp):
    name = os.path.join("Sample{}".format(sample), "Comp{:03}.hdf5".format(comp))
    return os.path.join(get_prefix(dataset), name)

def grad_cam(dataset, cv, on, is_cbam, df, dir="./attention_maps"):
    df = df[(df["dataset"]==dataset) & (df["cv"]==cv)]
    timestamp = get_timestamp(dataset, is_cbam)
    data_keys = ["data", "label", "tdata", "fdata", 'fdata_new']
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    nii_sample = get_sample(dataset)

    if is_cbam:
        res = cbam.resnet_test3_234
    else:
        res = resnet.resnet_test2

    model = Resnet_sm_gradcam(res=res)
    weight = get_state_dict(torch.load(os.path.join("model/model{}".format(timestamp), "model_final_{}_{}_{:04}_sm.pth".format(dataset, cv, 300))))
    #if "li2_sm.weight" in weight:
        #del weight["li2_sm.weight"], weight["li2_sm.bias"]
    try:
        model.load_state_dict(weight)
    except Exception as e:
        print(e)
        model.load_state_dict(weight, strict=False)
    model.to(device=device)
    model.eval()
    hdf5 = Hdf5(dataset=dataset, is_train=False, num_source=cv, data_keys=data_keys, source_dir="patchList_new")
    #evaluator = gcam.Evaluator("attention_maps")
    model = gcam.inject(model, output_dir=dir, backend='gcam', layer='net_sm.layer4',
                        label='best', save_maps=False)

    #test_filenames = get_names_comp_list(df)
    with torch.no_grad():
        step_size = len(df)
        for i in range(step_size):
            sample = df["sample"].iloc[i]
            comp = df["comp"].iloc[i]
            decision = df["{}_{}".format("cbam" if is_cbam else "resnet", on)].iloc[i]
            name = get_name_sample_comp(dataset,sample, comp)
            img = hdf5.getDataDicByName(name)["data"]
            #tdata = hdf5.getDataDicByName(name)["tdata"]
            label = int(hdf5.getDataDicByName(name)["label"].squeeze())
            score_file = "test/result_mat/score_{}{}_{}_{}.mat".format(dataset, timestamp, "cbam" if is_cbam else "resnet", on)
            score = loadmat(score_file)
            score = score["s{}".format(sample)][0][comp-1]
            if decision == 0:
                score = 1-score


            print("{}/{} sample {} comp {} decision {} score {:.4f}".format(i+1, step_size, sample, comp, decision, score))
            img_shape = img.squeeze().shape

            img = torch.tensor(img, dtype=torch.float).to(device)
            #tdata = torch.tensor(tdata, dtype=torch.float).to(device)
            _ = model(img)
            mask = model.get_attention_map().squeeze()
            mask = torch.from_numpy(mask[label])
            #print(mask.max(), mask.min(), mask.shape)

            img = img.squeeze().cpu().detach().numpy()
            #print(img.max(), img.min(), img.shape)
            ni_img = nib.Nifti1Image(img, nii_sample.affine, nii_sample.header)

            new_mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=img_shape, mode="trilinear")
            new_mask = new_mask.squeeze().cpu().detach().numpy()
            #print(new_mask.max(), new_mask.min(), new_mask.shape)

            thr = 1e-5
            _m = (abs(img) >= thr).astype(np.int)
            new_mask = new_mask * _m

            ni_mask = nib.Nifti1Image(new_mask, nii_sample.affine, nii_sample.header)
            prefix = dir+'/{}_sample{:03}_comp{:03}_label{}/'.format(dataset, sample, comp, label)
            postfix = '_{}_sample{:03}_comp{:03}_label{}'.format(dataset, sample, comp, label)
            if not os.path.exists(prefix):
                os.mkdir(prefix)
            if is_cbam:
                nib.save(ni_img, os.path.join(prefix, 'img'+postfix+".nii.gz"))
            postfix += '_{}{}[{:.4f}]'.format("cbam" if is_cbam else "resnet", df["{}_{}".format("cbam" if is_cbam else "resnet", on)].iloc[i], score)
            nib.save(ni_mask, os.path.join(prefix, 'map'+postfix+".nii.gz"))

            #plot(mask)
            #plot(new_mask)
            #show_cam(img=img, mask=new_mask)

def grad_cam_ts(dataset, cv, on, is_cbam, df, cmap="Reds", dir="./attention_maps"):
    df = df[(df["dataset"]==dataset) & (df["cv"]==cv)]
    timestamp = get_timestamp(dataset, is_cbam)
    data_keys = ["data", "label", "tdata", "fdata", 'fdata_new']
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    #nii_sample = get_sample(dataset)

    if is_cbam:
        res = cbam.resnet_test3_234
    else:
        res = resnet.resnet_test2

    model = Resnet_ts_gradcam(res=res).to(device=device)
    cam = Grad_CAM_ts(model)

    weight = get_state_dict(torch.load(os.path.join("model/model{}".format(timestamp), "model_final_{}_{}_{:04}_ts.pth".format(dataset, cv, 300))))
    try:
        model.load_state_dict(weight)
    except Exception as e:
        print(e)
        model.load_state_dict(weight, strict=False)
    model.eval()
    hdf5 = Hdf5(dataset=dataset, is_train=False, num_source=cv, data_keys=data_keys, source_dir="patchList_new")

    step_size = len(df)
    for i in range(step_size):
        sample = df["sample"].iloc[i]
        comp = df["comp"].iloc[i]
        decision = df["{}_{}".format("cbam" if is_cbam else "resnet", on)].iloc[i]
        name = get_name_sample_comp(dataset,sample, comp)
        tdata = hdf5.getDataDicByName(name)["tdata"]
        label = int(hdf5.getDataDicByName(name)["label"].squeeze())
        score_file = "test/result_mat/score_{}{}_{}_{}.mat".format(dataset, timestamp, "cbam" if is_cbam else "resnet", on)
        score = loadmat(score_file)
        score = score["s{}".format(sample)][0][comp-1]
        if decision == 0:
            score = 1-score

        print("{}/{} sample {} comp {} decision {} score {:.4f}".format(i+1, step_size, sample, comp, decision, score))
        img_shape = tdata.squeeze().shape
        tdata = torch.tensor(tdata, dtype=torch.float).to(device)[0]
        outs = model(tdata).squeeze()
        map = cam.get_cam(tdata, torch.argmax(outs,dim=-1))

        tdata = tdata.squeeze().cpu().detach().numpy()
        x = range(tdata.shape[-1])
        points = np.array([x, tdata]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        fig, ax = plt.subplots(1, 1, figsize=(20,5))
        norm = plt.Normalize(0, 1)
        lc = LineCollection(segments, cmap=truncate_colormap(cmap,0.2,1.0), norm=norm)

        # Set the values used for colormapping
        lc.set_array(map)
        lc.set_linewidth(2)
        line = ax.add_collection(lc)
        #fig.colorbar(line, ax=ax)
        ax.set_xlim(0, tdata.shape[0])
        ax.set_ylim(-1.1, 1.1)

        prefix = dir+'/{}_sample{:03}_comp{:03}_label{}/'.format(dataset, sample, comp, label)
        postfix = '_{}_sample{:03}_comp{:03}_label{}'.format(dataset, sample, comp, label)
        if not os.path.exists(prefix):
            os.mkdir(prefix)
        postfix += '_{}{}[{:.4f}]'.format("cbam" if is_cbam else "resnet", df["{}_{}".format("cbam" if is_cbam else "resnet", on)].iloc[i], score)
        ax.title.set_text('tdata'+postfix+".png")
        #plt.savefig(os.path.join(prefix, 'tdata'+postfix+".png"), dpi=200)
        #plt.show()

class Grad_CAM_ts:
    def __init__(self,model):
        self.gradient = []
        self.model = model
        self.model.net_ts.layer4.register_forward_hook(self.forward_hook)
        self.model.net_ts.layer4.register_backward_hook(self.backward_hook)

    def forward_hook(self, _, input, output):
        self.forward_result = torch.squeeze(output)

    def backward_hook(self, _, grad_input, grad_output):
        self.backward_result = torch.squeeze(grad_output[0])

    def get_cam(self, x, t):
        # 자동으로 foward_hook 함수가 호출되고, self.forward_result에 관찰하려는 layer를 거친 output이 저장됩니다.
        outs = self.model(x).squeeze()

        # backward를 통해서 자동으로 backward_hook 함수가 호출되고, self.backward_result에 gradient가 저장됩니다.
        outs[t].backward(retain_graph=True)

        # gradient의 평균을 구합니다. (alpha_k^c)
        a_k = torch.mean(self.backward_result, dim=1, keepdim=True)

        # self.foward_result를 이용하여 Grad-CAM을 계산합니다.
        out = torch.sum(a_k * self.forward_result, dim=0).cpu()

        # normalize
        out = (out + torch.abs(out)) / 2
        out = out / torch.max(out)

        # Bilinear upsampling (14*14 -> 224*224)
        m = torch.nn.Upsample(scale_factor=16, mode='linear')
        return m(out.unsqueeze(0).unsqueeze(0)).detach().squeeze().numpy()


def resnet_cbam_comparsion():
    df = pd.read_csv("./test/result.csv")
    #print(df.head())
    on = "sm"
    #datasets = "hcp bcp mb6 std".split()
    #datasets = "bcp mb6 std".split()
    datasets = ["hcp"]
    samples = [1,4,13,28,37,45,46,60,90,91] #np.random.choice(np.arange(1,101), 10,replace=False)

    for dataset in datasets:
        for cv in range(1,6):
            print("dataset:{} cv:{}".format(dataset, cv))
            _df = df[df["sample"].isin(samples)]
            print(_df.head())
            grad_cam(dataset, cv, on, is_cbam=True, df=_df)
            grad_cam(dataset, cv, on, is_cbam=False, df=_df)

    #print(df[df["resnet_{}".format(on)] != df["cbam_{}".format(on)]][["dataset","cv","sample","comp","resnet_vote","cbam_vote","label"]].head())

def random_sample_correct_wrong(num_samples = 10, dataset="hcp"):
    df = pd.read_csv("./test/result.csv")
    print(df.columns)
    dfc = df.copy()
    #dataset = "bcp"
    df = df[df["dataset"] == dataset]
    # del df["dataset"]
    #df = df[["sample", "comp", "label", "fix", "cbam_sm", "resnet_sm", "cbam_sm", "resnet_sm", "cbam_sm", "resnet_sm"]]

    d1 = df[(df["fix"] != df["cbam_vote"]) & (df["cbam_vote"] == df["label"])]  # cbam이 잘한 케이스
    d2 = df[(df["fix"] != df["cbam_vote"]) & (df["cbam_vote"] != df["label"])]  # cbam이 못한 케이스

    print(d1.shape[0], d2.shape[0])
    # print(d1.head())
    # print(d2.head())
    ridx1 = np.array([*np.random.choice(d1[d1["label"] == 0].index, num_samples // 2, replace=False),
                      *np.random.choice(d1[d1["label"] == 1].index, num_samples // 2)])
    ridx2 = np.array([*np.random.choice(d2[d2["label"] == 0].index, num_samples // 2, replace=False),
                      *np.random.choice(d2[d2["label"] == 1].index, num_samples // 2)])
    """
    while True:
        rlist1 = np.random.choice(d1.shape[0], num_samples)
        rlist2 = np.random.choice(d2.shape[0], num_samples)

        #print(d1.iloc[rlist1][["sample", "comp", "label", "fix", "cbam_sm", "resnet_sm"]])
        #print(d2.iloc[rlist2][["sample", "comp", "label", "fix", "cbam_sm", "resnet_sm"]])
        ridx1 = d1.iloc[rlist1].index
        ridx2 = d2.iloc[rlist2].index
        print(dfc.iloc[ridx1][["sample", "comp", "label", "fix", "cbam_sm", "resnet_sm"]])
        print(dfc.iloc[ridx2][["sample", "comp", "label", "fix", "cbam_sm", "resnet_sm"]])

        if dfc.iloc[ridx1]["label"].sum() > num_samples*2//5 and dfc.iloc[ridx2]["label"].sum() > num_samples*2//5:
            break
    """
    timestamp = datetime.today().strftime("_%Y%m%d%H%M%S")
    dir = "./attention_maps/random_sample_{}/".format(timestamp)
    if not os.path.exists(dir):
        os.mkdir(dir)

    _df1 = dfc.iloc[ridx1]
    _df2 = dfc.iloc[ridx2]
    _df1.to_csv(os.path.join(dir, "cbam_correct.csv"))
    _df2.to_csv(os.path.join(dir, "cbam_wrong.csv"))
    print(_df1)
    print(_df2)
    dir1 = dir + "cbam_correct/"
    dir2 = dir + "cbam_wrong/"
    for cv in range(1, 6):
        print("dataset:{} cv:{}".format(dataset, cv))
        grad_cam(dataset, cv, "sm", is_cbam=True, df=_df1, dir=dir1)
        grad_cam(dataset, cv, "sm", is_cbam=False, df=_df1, dir=dir1)
        grad_cam(dataset, cv, "sm", is_cbam=True, df=_df2, dir=dir2)
        grad_cam(dataset, cv, "sm", is_cbam=False, df=_df2, dir=dir2)
        grad_cam_ts(dataset, cv, "ts", is_cbam=True, df=_df1, dir=dir1)
        grad_cam_ts(dataset, cv, "ts", is_cbam=False, df=_df1, dir=dir1)
        grad_cam_ts(dataset, cv, "ts", is_cbam=True, df=_df2, dir=dir2)
        grad_cam_ts(dataset, cv, "ts", is_cbam=False, df=_df2, dir=dir2)

    return timestamp

def random_sample_correct_wrong_all(num_samples = 10, dataset="hcp"):
    df = pd.read_csv("./test/result.csv")
    print(df.columns)
    dfc = df.copy()
    df = df[df["dataset"] == dataset]
    # del df["dataset"]
    label = 0
    #df = df[["sample", "comp", "label", "fix", "cbam_sm", "resnet_sm", "cbam_ts", "resnet_ts"]]

    d1 = df[(df["fix"] == df["label"]) & (df["cbam_vote"] == df["label"]) & (df["cbam_smts"] == df["label"]) & (df["cbam_sm"] == df["label"]) & (df["cbam_ts"] == df["label"]) & (label == df["label"])]  # 둘다 잘한 케이스
    #d1 = df[(df["fix"] == df["cbam_sm"]) & (df["cbam_sm"] == df["label"]) & (df["cbam_ts"] == df["label"])]
    #d2 = df[(df["fix"] != df["cbam_sm"]) & (df["cbam_sm"] != df["label"]) & (df["cbam_ts"] != df["label"])]  # 둘다 못한 케이스

    #print(d1.shape[0], d2.shape[0])
    # print(d1.head())
    # print(d2.head())

    #ridx1 = np.array([*np.random.choice(d1[d1["label"] == 0].index, num_samples // 2), *np.random.choice(d1[d1["label"] == 1].index, num_samples // 2)])
    try:
        ridx1 = np.random.choice(d1[d1["label"] == label].index, num_samples, replace=False)
    except Exception:
        ridx1 = d1[d1["label"] == label].index
    #while True:
        #rlist1 = np.random.choice(d1.shape[0], num_samples)
        #rlist2 = np.random.choice(d2.shape[0], num_samples)

        #print(d1.iloc[rlist1][["sample", "comp", "label", "fix", "cbam_sm", "resnet_sm"]])
        #print(d2.iloc[rlist2][["sample", "comp", "label", "fix", "cbam_sm", "resnet_sm"]])
        #ridx1 = d1.iloc[rlist1].index
        #ridx2 = d2.iloc[rlist2].index
        #print(dfc.iloc[ridx1][["sample", "comp", "label", "fix", "cbam_sm", "resnet_sm"]])
        #print(dfc.iloc[ridx2][["sample", "comp", "label", "fix", "cbam_sm", "resnet_sm"]])

        #if dfc.iloc[ridx1]["label"].sum() > num_samples//3:# and dfc.iloc[ridx2]["label"].sum() > 3:
        #    break
    timestamp = datetime.today().strftime("_%Y%m%d%H%M%S")
    dir = "./attention_maps/random_sample_{}/".format(timestamp)
    if not os.path.exists(dir):
        os.mkdir(dir)

    _df1 = dfc.iloc[ridx1]
    _df1.to_csv(os.path.join(dir, "cbam_correct.csv"))
    print(_df1)
    #print(_df2)
    dir1 = dir + "cbam_correct/"
    for cv in range(1, 6):
        print("dataset:{} cv:{}".format(dataset, cv))
        grad_cam(dataset, cv, "sm", is_cbam=True, df=_df1, dir=dir1)
        grad_cam(dataset, cv, "sm", is_cbam=False, df=_df1, dir=dir1)
        #grad_cam(dataset, cv, "sm", is_cbam=True, df=_df2, dir=dir2)
        #grad_cam(dataset, cv, "sm", is_cbam=False, df=_df2, dir=dir2)
        grad_cam_ts(dataset, cv, "ts", is_cbam=True, df=_df1, dir=dir1)
        grad_cam_ts(dataset, cv, "ts", is_cbam=False, df=_df1, dir=dir1)
        #grad_cam_ts(dataset, cv, "ts", is_cbam=True, df=_df2, dir=dir2)
        #grad_cam_ts(dataset, cv, "ts", is_cbam=False, df=_df2, dir=dir2)

    return timestamp

def random_sample_choice(dataset, sample, comp):
    df = pd.read_csv("./test/result.csv")
    print(df.columns)
    dfc = df.copy()
    df = df[df["dataset"] == dataset]
    # del df["dataset"]
    label = 0
    #df = df[["sample", "comp", "label", "fix", "cbam_sm", "resnet_sm", "cbam_ts", "resnet_ts"]]

    d1 = None
    for i in range(len(sample)):
        if d1 is None:
            d1 = df[(df["sample"] == sample[i]) & (df["comp"] == comp[i])]
        else:
            d1 = pd.concat([d1, df[(df["sample"] == sample[i]) & (df["comp"] == comp[i])]], axis=0, ignore_index=True)

    timestamp = datetime.today().strftime("_%Y%m%d%H%M%S")
    dir = "./attention_maps/random_sample_{}/".format(timestamp)
    if not os.path.exists(dir):
        os.mkdir(dir)

    _df1 = d1
    _df1.to_csv(os.path.join(dir, "cbam_correct.csv"))
    print(_df1)
    #print(_df2)
    dir1 = dir + "cbam_correct/"
    for cv in range(1, 6):
        print("dataset:{} cv:{}".format(dataset, cv))
        grad_cam(dataset, cv, "sm", is_cbam=True, df=_df1, dir=dir1)
        grad_cam(dataset, cv, "sm", is_cbam=False, df=_df1, dir=dir1)
        #grad_cam(dataset, cv, "sm", is_cbam=True, df=_df2, dir=dir2)
        #grad_cam(dataset, cv, "sm", is_cbam=False, df=_df2, dir=dir2)
        grad_cam_ts(dataset, cv, "ts", is_cbam=True, df=_df1, dir=dir1)
        grad_cam_ts(dataset, cv, "ts", is_cbam=False, df=_df1, dir=dir1)
        #grad_cam_ts(dataset, cv, "ts", is_cbam=True, df=_df2, dir=dir2)
        #grad_cam_ts(dataset, cv, "ts", is_cbam=False, df=_df2, dir=dir2)

    return timestamp

# def plot_sm_ts(df, dir):
#     """
#     visualize all the nii image in dir
#     :param df:
#     :param dir:
#     :return:
#     """
#     for i in range(df.shape[0]):
#         dataset = df["dataset"].iloc[i]
#         sample = df["sample"].iloc[i]
#         comp = df["comp"].iloc[i]
#         label = df["label"].iloc[i]
#
#         overlays = get_nii_name(dataset, sample, comp, label, dir)
#         cbam_tdata, cbam_tmap, fdata = get_ts(dataset,sample,comp,is_cbam=True)
#         res_tdata, res_tmap, _ = get_ts(dataset,sample,comp,is_cbam=False)
#         print(overlays)
#         nii_input = nib.load(overlays[0])
#         nii_cbam = nib.load(overlays[1])
#         nii_resnet = nib.load(overlays[2])
#         overlay_input = np.asarray(nii_input.dataobj)
#         overlay_cbam = np.asarray(nii_cbam.dataobj)
#         overlay_resnet = np.asarray(nii_resnet.dataobj)
#         X,Y,Z = overlay_input.shape
#         #print(overlay_input.shape)
#
#         N = res_tdata.shape[-1]
#         T = get_sampling_time(dataset)
#         xf = fftfreq(N, T)[:N // 2]
#
#         ## Threshold
#         thre = 0.02
#         overlay_input[np.abs(overlay_input) < thre] = 0
#
#         #figure config
#         #struct = get_structure_from_sample(dataset, sample)
#         struct = get_structure(pad_dataset=dataset)
#         #plt.rcParams['savefig.facecolor'] = 'white'
#         fig = plt.figure(figsize=(16, 12))
#
#         bottom_pos, top_pos, left_pos, right_pos = 0.3,0.9,0.1,0.9
#
#         fig.tight_layout()
#         #fig.subplots_adjust(bottom=bottom_pos, top=top_pos, left=left_pos, right=right_pos)
#         gs = GridSpec(nrows=4, ncols=3, width_ratios=[1/Y, 1/Z, 1/Z], height_ratios=[5,2,2,1])
#         prefix = dir + '/{}_sample{:03}_comp{:03}_label{}/'.format(dataset, sample, comp, label)
#         postfix = 'plot_cbam_{}_sample{:03}_comp{:03}_label{}_sm{}_ts{}'.format(dataset, sample, comp, label, df["cbam_sm"].iloc[i], df["cbam_ts"].iloc[i])
#         plot_3dmap_overlays(fig, gs, struct, overlay_input, overlay_cbam, coord=get_max_coor(overlay_cbam), df=df.iloc[i], is_cbam=True)
#         plot_ts(fig, gs, cbam_tdata, cbam_tmap)
#         plot_fdata(fig, gs, fdata, xf)
#         plt.tight_layout()
#         plt.savefig(os.path.join(prefix,postfix+".png"), edgecolor='none', pad_inches=0.5, bbox_inches='tight')
#         #plt.show()
#
#         fig = plt.figure(figsize=(16, 12))
#         fig.tight_layout()
#         #fig.subplots_adjust(bottom=bottom_pos, top=top_pos, left=left_pos, right=right_pos)
#         gs = GridSpec(nrows=4, ncols=3, width_ratios=[1/Y, 1/Z, 1/Z], height_ratios=[5,2,2,1])
#         postfix = 'plot_resnet_{}_sample{:03}_comp{:03}_label{}_sm{}_ts{}'.format(dataset, sample, comp, label, df["resnet_sm"].iloc[i], df["resnet_ts"].iloc[i])
#         plot_3dmap_overlays(fig, gs, struct, overlay_input, overlay_resnet, coord=get_max_coor(overlay_resnet), df=df.iloc[i], is_cbam=False)
#         plot_ts(fig, gs, res_tdata, res_tmap)
#         plot_fdata(fig, gs, fdata, xf)
#         plt.tight_layout()
#         plt.savefig(os.path.join(prefix,postfix+".png"), edgecolor='none', pad_inches=0.5, bbox_inches='tight')
#         #plt.show()

def get_ts(dataset, sample, comp, is_cbam):
    df = pd.read_csv("./test/result.csv")
    df = df[(df["dataset"]==dataset)&(df["sample"]==sample)&(df["comp"]==comp)]
    cv = df["cv"].iloc[0]
    on = "ts"
    timestamp = get_timestamp(dataset, is_cbam)
    data_keys = ["data", "label", "tdata", "fdata", 'fdata_new']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # nii_sample = get_sample(dataset)

    if is_cbam:
        res = cbam.resnet_test3_234
    else:
        res = resnet.resnet_test2

    model = Resnet_ts_gradcam(res=res).to(device=device)
    cam = Grad_CAM_ts(model)

    weight = get_state_dict(torch.load(
        os.path.join("model/model{}".format(timestamp), "model_final_{}_{}_{:04}_ts.pth".format(dataset, cv, 300))))
    try:
        model.load_state_dict(weight)
    except Exception as e:
        print(e)
        model.load_state_dict(weight, strict=False)
    model.eval()
    hdf5 = Hdf5(dataset=dataset, is_train=False, num_source=cv, data_keys=data_keys, source_dir="patchList_new")

    step_size = len(df)
    for i in range(step_size):
        sample = df["sample"].iloc[i]
        comp = df["comp"].iloc[i]
        decision = df["{}_{}".format("cbam" if is_cbam else "resnet", on)].iloc[i]
        name = get_name_sample_comp(dataset, sample, comp)
        tdata = hdf5.getDataDicByName(name)["tdata"]
        fdata = hdf5.getDataDicByName(name)["fdata" if dataset=="bcp" else "fdata_new"]
        label = int(hdf5.getDataDicByName(name)["label"].squeeze())
        score_file = "test/result_mat/score_{}{}_{}_{}.mat".format(dataset, timestamp, "cbam" if is_cbam else "resnet",on)
        score = loadmat(score_file)
        score = score["s{}".format(sample)][0][comp - 1]
        if decision == 0:
            score = 1 - score

        print("{}/{} sample {} comp {} decision {} score {:.4f}".format(i + 1, step_size, sample, comp, decision, score))
        tdata = torch.tensor(tdata, dtype=torch.float).to(device)[0]
        fdata = torch.tensor(fdata, dtype=torch.float).to(device)[0] # 이럴필요는 없는데 그냥 편의상
        outs = model(tdata).squeeze()
        tmap = cam.get_cam(tdata, torch.argmax(outs, dim=-1))
        tdata = tdata.squeeze().cpu().detach().numpy()
        fdata = fdata.squeeze().cpu().detach().numpy()

        return tdata, tmap, fdata

# def plot_ts(fig, gs, tdata, tmap, cmap="Reds"):
#     ax = fig.add_subplot(gs[-3,:])
#     ax.set_facecolor('white')
#     norm = plt.Normalize(0, 1)
#     x = range(tdata.shape[-1])
#     points = np.array([x, tdata]).T.reshape(-1, 1, 2)
#     segments = np.concatenate([points[:-1], points[1:]], axis=1)
#
#     lc = LineCollection(segments, cmap=truncate_colormap(cmap, 0.4, 1.0), norm=norm)
#
#     # Set the values used for colormapping
#     lc.set_array(tmap)
#     lc.set_linewidth(4)
#     line = ax.add_collection(lc)
#
#     # cb2 = fig.colorbar(line, ax=ax, pad=0.05, fraction=0.015, ticks=[0,.5,1], aspect=10)
#     # cb2.ax.tick_params(labelsize=30)
#     # ax.plot(range(1,len(tdata)+1), tdata, '#FC8868', linewidth=4)
#     ax.set_xlim(0, tdata.shape[0])
#     ax.set_ylim(-1.1, 1.1)
#     # ax.set_xlabel("Time index", fontsize=40)
#     # ax.set_ylabel("Response", fontsize=40)
#     # ax.xaxis.set_tick_params(labelsize=40)
#     # ax.yaxis.set_tick_params(labelsize=40)
#     ax.axes.xaxis.set_ticklabels([])
#     ax.axes.yaxis.set_ticklabels([])


def plot_fdata(fig, gs, fdata, xtick):
    ax = fig.add_subplot(gs[2,:])
    x = range(fdata.shape[-1])
    xtick = np.array(xtick, dtype=np.float) * 100
    ax.plot(xtick, fdata, 'c')
    ax.set_xlim(0, xtick[-1])
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0,1])
    ax.set_xlabel("Frequency (Hz/100)", fontsize=20)
    ax.set_ylabel("Power", fontsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)

def get_sampling_time(dataset):
    if dataset == "bcp":
        ts = 0.8
    elif dataset == "hcp":
        ts = 0.73
    elif dataset == "mb6":
        ts = 1.3
    elif dataset == "std":
        ts = 3
    else:
        ts = None

    return ts

def plot_3dmap_overlays(fig, axes, struct, img, attention_map, coord, df, is_cbam):
    #csfont = {'family':'serif','serif':['Times']}
    rc('font', **{'family':'serif','sans-serif':['Times']})
    rc("text", usetex=True)
    font = mpl.font_manager.FontProperties()
    font.set_family("Times New Roman")
    font.set_size(30)
    font2 = mpl.font_manager.FontProperties()
    font2.set_family("Times New Roman")
    font2.set_size(35)

    cnn = 'cbam' if is_cbam else 'resnet'
    X, Y, Z = attention_map.shape
    x, y, z = coord
    print("coord : ", x, y, z)

    img = img.copy()
    img_clip = 0.3
    img = clip_3dimg(img, img_clip)
    print("img min {}, max {}".format(img.min(), img.max())  )
    img_maxrange = img_clip
    img_minrange = -img_clip

    img_x = np.rot90(img[x, :, :])
    img_y = np.rot90(img[:, y, :])
    img_z = np.rot90(img[:, :, z])
    img_cmap = copy(plt.cm.get_cmap("seismic"))
    img_cmap.set_bad(alpha=0)
    img[img==0] = np.nan

    map_x = np.rot90(attention_map[x, :, :])
    map_y = np.rot90(attention_map[:, y, :])
    map_z = np.rot90(attention_map[:, :, z])
    map_cmap = "Greens"
    #map_cmap = "hot"

    _X, _Y, _Z = struct.shape
    #struct = np.array(Image.fromarray(struct).resize(img.shape))
    struct_cmap = "Greys"
    _x = round(x*(_X/X))
    _y = round(y*(_Y/Y))
    _z = round(z*(_Z/Z))
    struct_x = np.rot90(struct[_x, :, :])
    struct_y = np.rot90(struct[:, _y, :])
    struct_z = np.rot90(struct[:, :, _z])
    struct_x = np.array(Image.fromarray(struct_x).resize(map_x.T.shape))
    struct_y = np.array(Image.fromarray(struct_y).resize(map_y.T.shape))
    struct_z = np.array(Image.fromarray(struct_z).resize(map_z.T.shape))

    if attention_map.shape != struct.shape:
        print("shape is not match {} {}".format(attention_map.shape, struct.shape))

    map_maxrange = 1#abs(attention_map).max()
    map_minrange = 0
    struct_maxrange = struct.max()
    struct_minrange = struct.min()

    linecolor = "darkviolet"
    linewidth = 5
    margin = 5
    alpha1 = 0.4
    alpha2 = 1
    #title = "{} sample{} comp{}".format(df["dataset"], df["sample"], df["comp"])
    #fig.suptitle(title, horizontalalignment='center', verticalalignment="center",x=0.5, y=0.45)
    #info = "{}\nLabel  : {}\nFIX    : {}\nCNN_sm : {}\nCNN_ts : {}\n".format(cnn.upper(), df["fix"], df["label"], df["{}_sm".format(cnn)], df["{}_ts".format(cnn)])
    mapping_str = {0:"noise", 1:"signal"}
    info = r"$Label$ : %s, $FIX$ : %s, $CNN_{vote}$ : %s, $CNN_{sm+ts}$ : %s, $CNN_{sm}$ : %s, $CNN_{ts}$ : %s" % (mapping_str[df["label"]], mapping_str[df["fix"]], mapping_str[df["{}_vote".format(cnn)]], mapping_str[df["{}_smts".format(cnn)]], mapping_str[df["{}_sm".format(cnn)]], mapping_str[df["{}_ts".format(cnn)]])
    fig.suptitle(info, horizontalalignment='center', x=0.5, y=0.08, fontproperties=font2)
    fig.patch.set_facecolor((1,0,0,0))
    ax1 = fig.add_subplot(axes[0, 1])
    ax1.set_title('Frontal plane', fontproperties=font)

    ax1.imshow(struct_y, cmap=struct_cmap, vmin=struct_minrange, vmax=struct_maxrange, interpolation='none')
    ax1.imshow(img_y, cmap=img_cmap, vmin=img_minrange, vmax=img_maxrange, alpha=alpha2, interpolation='none')
    ax1.imshow(map_y, cmap=map_cmap, vmin=map_minrange, vmax=map_maxrange, alpha=alpha1, interpolation='none')
    draw_line(ax1, x, z, X, Z, margin, linecolor, linewidth)
    #ax1.vlines(x, 0, Z - z - margin, colors=linecolor, linewidth=linewidth)
    #ax1.vlines(x, Z - z + margin, Z - 1, colors=linecolor, linewidth=linewidth)
    #ax1.hlines(Z - z, 0, x - margin, colors=linecolor, linewidth=linewidth)
    #ax1.hlines(Z - z, x + margin, X - 1, colors=linecolor, linewidth=linewidth)
    ax1.axis('off')
    ax1.tick_params(axis='both', which='both', length=0)

    ax2 = fig.add_subplot(axes[0,2])
    ax2.set_title('Sagittal plane', fontproperties=font)
    ax2.imshow(struct_x, cmap=struct_cmap, vmin=struct_minrange, vmax=struct_maxrange, interpolation='none')
    im = ax2.imshow(img_x, cmap=img_cmap, vmin=img_minrange, vmax=img_maxrange, alpha=alpha2, interpolation='none')
    overlay = ax2.imshow(map_x, cmap=map_cmap, vmin=map_minrange, vmax=map_maxrange, alpha=alpha1, interpolation='none')
    draw_line(ax2, y, z, Y, Z, margin, linecolor, linewidth)
    #ax2.vlines(y, 0, Z - z - margin, colors=linecolor, linewidth=linewidth)
    #ax2.vlines(y, Z - z + margin, Z - 1, colors=linecolor, linewidth=linewidth)
    #ax2.hlines(Z - z, 0, y - margin, colors=linecolor, linewidth=linewidth)
    #ax2.hlines(Z - z, y + margin, Y - 1, colors=linecolor, linewidth=linewidth)
    ax2.axis('off')
    ax2.tick_params(axis='both', which='both', length=0)

    #divider = make_axes_locatable(ax2)
    #cax1 = divider.new_horizontal(size="5%", pad=1.5)
    #cax2 = divider.new_horizontal(size="5%", pad=0)
    #fig.add_axes(cax1)
    #fig.add_axes(cax2)

    #cb2 = plt.colorbar(im, pad=0, cax=cax2, ticks=[-img_clip,0,img_clip])
    #cb1 = plt.colorbar(overlay, pad=1, cax=cax1, ticks=[0,.5,1])
    #cb1.ax.yaxis.set_ticks_position('left')
    #cb1.ax.tick_params(labelsize=30)
    #cb2.ax.tick_params(labelsize=30)

    ax3 = fig.add_subplot(axes[0,0])
    ax3.set_title('Horizontal plane', fontproperties=font)
    ax3.imshow(struct_z, cmap=struct_cmap, vmin=struct_minrange, vmax=struct_maxrange, interpolation='none')
    ax3.imshow(img_z, cmap=img_cmap, vmin=img_minrange, vmax=img_maxrange, alpha=alpha2, interpolation='none')
    ax3.imshow(map_z, cmap=map_cmap, vmin=map_minrange, vmax=map_maxrange, alpha=alpha1, interpolation='none')

    draw_line(ax3, x, y, X, Y, margin, linecolor, linewidth)
    #ax3.vlines(x, 0, Y - y - margin, colors=linecolor, linewidth=linewidth)
    #ax3.vlines(x, Y - y + margin, Y - 1, colors=linecolor, linewidth=linewidth)
    #ax3.hlines(Y - y, 0, x - margin, colors=linecolor, linewidth=linewidth)
    #ax3.hlines(Y - y, x + margin, X - 1, colors=linecolor, linewidth=linewidth)
    ax3.axis('off')
    ax3.tick_params(axis='both', which='both', length=0)

def plot_slice(df, dir):
    """
    visualize all the nii image in dir
    :param df:
    :param dir:
    :return:
    """
    #mpl.rcParams['text.color'] = "white"
    for i in range(df.shape[0]):
        dataset = df["dataset"].iloc[i]
        sample = df["sample"].iloc[i]
        comp = df["comp"].iloc[i]
        label = df["label"].iloc[i]

        overlays = get_nii_name(dataset, sample, comp, label, dir)
        cbam_tdata, cbam_tmap, fdata = get_ts(dataset,sample,comp,is_cbam=True)
        res_tdata, res_tmap, _ = get_ts(dataset,sample,comp,is_cbam=False)
        print(overlays)
        nii_input = nib.load(overlays[0])
        nii_cbam = nib.load(overlays[1])
        nii_resnet = nib.load(overlays[2])
        overlay_input = np.asarray(nii_input.dataobj)
        overlay_cbam = np.asarray(nii_cbam.dataobj)
        overlay_resnet = np.asarray(nii_resnet.dataobj)
        X,Y,Z = overlay_input.shape
        #print(overlay_input.shape)

        N = res_tdata.shape[-1]
        T = get_sampling_time(dataset)
        xf = fftfreq(N, T)[:N // 2]

        ## Threshold
        thre = 0.02
        overlay_input[np.abs(overlay_input) < thre] = 0

        #figure config
        #struct = get_structure_from_sample(dataset, sample)
        struct = get_structure(pad_dataset=dataset)
        plt.rcParams['savefig.facecolor'] = 'white'
        fig = plt.figure(figsize=(34, 26))
        # gs[:-2, :].set_facecolor('black')
        fig.tight_layout()
        # fig.subplots_adjust(bottom=bottom_pos, top=top_pos, left=left_pos, right=right_pos)
        gs = GridSpec(nrows=6, ncols=8, width_ratios=[1, 1, 1, 1, 1, 1, 1, 0.15], height_ratios=[5, 5, 1.5, 4, 1.5, 3])
        gs.update(hspace=0.25)
        gs.update(left=0.1, right=0.9)
        x1, y1 = (0.1, 0.53)
        rect = Rectangle((x1, y1), 0.87 - x1, 0.9 - y1, facecolor="#190000", edgecolor='none',
                         transform=fig.transFigure, zorder=-1)
        fig.patches.append(rect)
        #gs[:-2, :].set_facecolor('black')
        prefix = dir + '/{}_sample{:03}_comp{:03}_label{}/'.format(dataset, sample, comp, label)
        postfix = 'plot_cbam_{}_sample{:03}_comp{:03}_label{}_sm{}_ts{}'.format(dataset, sample, comp, label, df["cbam_sm"].iloc[i], df["cbam_ts"].iloc[i])
        plot_3dmap_overlays_slice(fig, gs, struct, overlay_input, overlay_cbam, coord=get_max_coor(overlay_cbam), df=df.iloc[i], is_cbam=True)
        plot_ts(fig, gs, cbam_tdata, cbam_tmap)
        #plot_fdata(fig, gs, fdata, xf)
        plt.tight_layout()
        plt.savefig(os.path.join(prefix,postfix+".png"), edgecolor='none', pad_inches=0.3, bbox_inches='tight')
        plt.savefig(os.path.join(prefix,postfix+".pdf"), edgecolor='none', pad_inches=0.3, bbox_inches='tight')
        #plt.show()

        fig = plt.figure(figsize=(34, 26))
        # gs[:-2, :].set_facecolor('black')
        fig.tight_layout()
        # fig.subplots_adjust(bottom=bottom_pos, top=top_pos, left=left_pos, right=right_pos)
        gs = GridSpec(nrows=6, ncols=8, width_ratios=[1, 1, 1, 1, 1, 1, 1, 0.15], height_ratios=[5, 5, 1.5, 4, 1.5, 3])
        gs.update(hspace=0.25)
        gs.update(left=0.1, right=0.9)
        x1, y1 = (0.1, 0.53)
        rect = Rectangle((x1, y1), 0.87 - x1, 0.9 - y1, facecolor="#190000", edgecolor='none',
                         transform=fig.transFigure, zorder=-1)
        fig.patches.append(rect)
        postfix = 'plot_resnet_{}_sample{:03}_comp{:03}_label{}_sm{}_ts{}'.format(dataset, sample, comp, label, df["resnet_sm"].iloc[i], df["resnet_ts"].iloc[i])
        plot_3dmap_overlays_slice(fig, gs, struct, overlay_input, overlay_resnet, coord=get_max_coor(overlay_resnet), df=df.iloc[i], is_cbam=False)
        plot_ts(fig, gs, res_tdata, res_tmap)
        #plot_fdata(fig, gs, fdata, xf)
        plt.tight_layout()
        plt.savefig(os.path.join(prefix,postfix+".png"), edgecolor='none', pad_inches=0.3, bbox_inches='tight')
        plt.savefig(os.path.join(prefix,postfix+".pdf"), edgecolor='none', pad_inches=0.3, bbox_inches='tight')
        #plt.show()
        print("thre : ",thre)
        # return


def plot_3dmap_overlays_slice(fig, axes, struct, img, attention_map, coord, df, is_cbam):
    #csfont = {'family':'serif','serif':['Times']}
    rc('font', **{'family':'serif','sans-serif':['Times']})
    rc("text", usetex=True)
    font = mpl.font_manager.FontProperties()
    font.set_family("Times New Roman")
    font.set_size(30)
    font2 = mpl.font_manager.FontProperties()
    font2.set_family("Times New Roman")
    font2.set_size(35)

    cnn = 'cbam' if is_cbam else 'resnet'
    X, Y, Z = attention_map.shape
    x, y, z = coord
    print("coord : ", x, y, z)

    img = img.copy()
    img_clip = 0.3
    img = clip_3dimg(img, img_clip)
    print("img min {}, max {}".format(img.min(), img.max())  )
    img_maxrange = img_clip
    img_minrange = -img_clip

    img_cmap = truncate_colormap(copy(plt.cm.get_cmap("seismic")), 0, 1)
    img_cmap.set_bad(alpha=0)
    img[img==0] = np.nan


    map_cmap = truncate_colormap(copy(plt.cm.get_cmap("afmhot")), 0.2,1)
    #map_cmap = "hot"

    _X, _Y, _Z = struct.shape
    #struct = np.array(Image.fromarray(struct).resize(img.shape))
    struct_cmap = truncate_colormap(copy(plt.cm.get_cmap("gray")), 0, 1)


    if attention_map.shape != struct.shape:
        print("shape is not match {} {}".format(attention_map.shape, struct.shape))

    map_maxrange = 1 #abs(attention_map).max()
    map_minrange = 0
    struct_maxrange = struct.max()
    struct_minrange = struct.min()

    alpha1 = 0.25
    alpha2 = 1

    fig.patch.set_facecolor((1,0,0,0))
    # fig.patch.set_facecolor("#190000")
    for i in range(14):
        z = int(Z/(14+4)*(i+4))
        img_z = np.fliplr(np.rot90(img[:, :, z]))
        map_z = np.fliplr(np.rot90(attention_map[:, :, z]))
        _z = round(z * (_Z / Z))
        struct_z = np.fliplr(np.rot90(struct[:, :, _z]))
        struct_z = np.array(Image.fromarray(struct_z).resize(map_z.T.shape))

        ax = fig.add_subplot(axes[i//7, i%7])
        ax.imshow(struct_z, cmap=struct_cmap, vmin=struct_minrange, vmax=struct_maxrange, interpolation='bilinear', aspect='equal')
        ax_img = ax.imshow(img_z, cmap=img_cmap, vmin=img_minrange, vmax=img_maxrange, alpha=alpha2, interpolation='bilinear', aspect='equal')
        ax_map = ax.imshow(map_z, cmap=map_cmap, vmin=map_minrange, vmax=map_maxrange, alpha=alpha1, interpolation='bilinear', aspect='equal')

        #draw_line(ax, x, y, X, Y, margin, linecolor, linewidth)
        ax.axis('off')
        ax.tick_params(axis='both', which='both', length=0)

    ax_cb1 = fig.add_subplot(axes[0, -1])
    cb = fig.colorbar(ax_img, cax=ax_cb1, ticks=[-img_clip,0,img_clip])
    cb.ax.tick_params(labelsize=50)

    ax_cb2 = fig.add_subplot(axes[1, -1])
    cb = fig.colorbar(ax_map, cax=ax_cb2, ticks=[0,.5,1])
    cb.ax.tick_params(labelsize=50)

    # plot result table
    mapping_str = {0:"Noise", 1:"Signal"}
    ax = fig.add_subplot(axes[-1, :])
    ax.axis('tight')
    ax.axis('off')
    superscript = {"cbam":"AM", "resnet":"Res"}
    column_labels = ["Human experts", "FIX", r"$CNN^{%s}_{sm}$"%superscript[cnn], r"$CNN^{%s}_{ts}$"%superscript[cnn], r"$CNN^{%s}_{sm+ts}$"%superscript[cnn], r"$CNN^{%s}_{vote}$"%superscript[cnn]]
    column_keys = ["label", "fix", "{}_sm".format(cnn), "{}_ts".format(cnn), "{}_smts".format(cnn), "{}_vote".format(cnn)]
    cell_text = [[mapping_str[df[key]] for key in column_keys]]
    t1 = ax.table(cellText=cell_text,colLabels=column_labels,loc="center",cellLoc='center')
    t1.auto_set_font_size(False)
    t1.set_fontsize(45)
    t1.scale(1, 7)


def plot_ts(fig, gs, tdata, tmap):
    cmap = truncate_colormap("Reds", minval=0.0, maxval=0.7)
    ax = fig.add_subplot(gs[-3,:-1])
    ax.set_facecolor('white')
    # norm = plt.Normalize(0, 1)
    X = np.arange(tdata.shape[-1])
    Y = np.array([-1.1, 1.1])
    yarr = np.vstack((tmap,tmap))
    # points = np.array([x, tdata]).T.reshape(-1, 1, 2)
    # segments = np.concatenate([points[:-1], points[1:]], axis=1)
    #
    # lc = LineCollection(segments, cmap=truncate_colormap(cmap, 0.4, 1.0), norm=norm)

    # Set the values used for colormapping
    # lc.set_array(tmap)
    # lc.set_linewidth(4)
    # line = ax.add_collection(lc)


    # cb2 = fig.colorbar(line, ax=ax, pad=0.05, fraction=0.015, ticks=[0,.5,1], aspect=10)
    # ax.imshow(yarr, extent=(1,len(tdata)+1,-1.1,1.1), cmap=cmap)

    # cb2 = fig.colorbar(pl, ax=ax, pad=0.05, fraction=0.015, ticks=[0,.5,1], aspect=10)
    # cb2.ax.tick_params(labelsize=30)

    pcm = ax.pcolormesh(X, Y, yarr, cmap=cmap, vmin=0, vmax=1)
    ax.plot(range(1,len(tdata)+1), tdata, "navy", linewidth=4)
    ax.set_xlim(0, tdata.shape[0])
    ax.set_ylim(-1.1, 1.1)
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    ax1 = fig.add_subplot(gs[-3,-1])
    cb = fig.colorbar(pcm, cax=ax1, fraction=0.015, ticks=[0,.5,1], shrink=0.6)
    cb.ax.tick_params(labelsize=50)


def draw_line(ax, x, y, X, Y, margin, linecolor, linewidth):
    if Y - y - margin > 1:
        ax.vlines(x, 1, Y - y - margin, colors=linecolor, linewidth=linewidth)
    if Y - y + margin < Y-1:
        ax.vlines(x, Y - y + margin, Y - 1, colors=linecolor, linewidth=linewidth)
    if x - margin > 1:
        ax.hlines(Y - y, 1, x - margin, colors=linecolor, linewidth=linewidth)
    if x + margin < X - 1:
        ax.hlines(Y - y, x + margin, X - 1, colors=linecolor, linewidth=linewidth)

def imresize(img, new_shape):
    im = Image.fromarray(img)
    return np.array(im.resize(new_shape, Image.BICUBIC))

def get_max_coor(array):
    """
    return argmax coordinate
    :param array: 3d-array
    :return: 3d indexs which has maxmum output
    """
    return np.unravel_index(np.argmax(array, axis=None), array.shape)


def get_nii_name(dataset, sample, comp, label, dir):
    """
    get nii names
    :param dataset:
    :param sample:
    :param comp:
    :param label:
    :param dir:
    :return: img, map_cbam, map_resnet
    """
    dir = os.path.join(dir, "{}_sample{:03d}_comp{:03d}_label{}".format(dataset,sample,comp,label))
    file_list = sorted([file for file in os.listdir(dir) if ".nii.gz" in file])

    return [os.path.join(dir, file_list[i]) for i in range(3)]

def get_structure(pad_dataset):
    sample = nib.load("data/Resliced_mni_2.nii")
    sample = np.asarray(sample.dataobj)
    sample = remove_pad(sample, dataset=pad_dataset)
    return sample

def get_structure_from_sample(dataset, sample):
    if dataset == "hcp":
        rootdir = '/home/user/mailab_nas/heo/Denoising/data/HCP_hp2000_All_FIX/'
    elif dataset == "std":
        rootdir = '/home/user/mailab_nas/heo/Denoising/data/WhII_Standard_FIX/'
    elif dataset == "mb6":
        rootdir = '/home/user/mailab_nas/heo/Denoising/data/WhII_MB6_FIX/'
    elif dataset == "bcp":
        rootdir = '/home/user/mailab_nas/heo/Denoising/data/BCP_for_FIX/'
    else:
        raise ValueError("Unknown dataset : {}".format(dataset))

    sample_dir = os.path.join(rootdir, "Sample{}/reg/example_func.nii.gz".format(sample))
    sample = nib.load(sample_dir)
    sample = np.asarray(sample.dataobj)
    sample = remove_pad(sample)
    sample_4q = int(sample.max() * 0.5)
    sample[sample>sample_4q] = sample_4q
    sample = (sample-sample.min())/(sample.max()-sample.min())

    return sample

def remove_pad(array, dataset = "hcp"):
    X, Y, Z = array.shape
    thre = 1e-6
    if dataset == "bcp":
        margin = 1  # -1
        bias = -1  # 1
    elif dataset == "hcp":
        margin = -1
        bias = 1
    else:
        margin = 0
        bias = 0

    for i in range(X):
        if array[i,:,:].max() > thre:
            x1 = i - 1 - margin
            break

    for i in range(X-1,0,-1):
        if array[i,:,:].max() > thre:
            x2 = i + 1 + margin
            break

    for i in range(Y):
        if array[:,i,:].max() > thre:
            y1 = i - 1 - margin
            break

    for i in range(Y-1,0,-1):
        if array[:,i,:].max() > thre:
            y2 = i + 1 + margin
            break


    for i in range(Z):
        if array[:,:,i].max() > thre:
            z1 = i - 1 + bias
            break

    for i in range(Z-1,0,-1):
        if array[:,:,i].max() > thre:
            z2 = i + 1 - bias
            break
    return array[x1:x2+1,y1:y2+1,z1:z2+1]

def add_ts_plot():
    dir = "./attention_maps/random_5_5sample"
    dir1 = dir + "/cbam_correct/"
    dir2 = dir + "/cbam_wrong/"
    dataset = "hcp"
    _df1 = pd.read_csv(os.path.join(dir, "cbam_correct.csv"))
    _df2 = pd.read_csv(os.path.join(dir, "cbam_wrong.csv"))
    for cv in range(1, 6):
        grad_cam_ts(dataset, cv, "ts", is_cbam=True, df=_df1, dir=dir1)
        grad_cam_ts(dataset, cv, "ts", is_cbam=False, df=_df1, dir=dir1)
        grad_cam_ts(dataset, cv, "ts", is_cbam=True, df=_df2, dir=dir2)
        grad_cam_ts(dataset, cv, "ts", is_cbam=False, df=_df2, dir=dir2)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    cmap = plt.get_cmap(cmap)
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def clip_3dimg(img, thre):
    arr = img.copy()
    arr[arr>thre] = thre
    arr[arr<-thre] = -thre
    return arr



if __name__ == '__main__':
    #description = ""
    #timestamp = random_sample_choice(dataset="hcp", sample=[1, 5, 43, 10, 5, 89, 25, 16, 97, 61, 33], comp=[40, 58, 69, 78, 158, 184, 88, 60, 131, 15, 77])
    # timestamp = random_sample_correct_wrong(40, dataset="bcp")
    # timestamps = [timestamp]
    #timestamp = random_sample_correct_wrong(100, dataset="hcp")
    #timestamps = [*timestamps, timestamp]
    #timestamp = random_sample_correct_wrong_all(300, dataset="hcp")
    #timestamps = [timestamp]
    #timestamp = random_sample_correct_wrong_all(30, dataset="bcp")
    #timestamps = [*timestamps, timestamp]
    #timestamps = ["_20201118183419", "_20201130160213", "_20201201103906"]
    #timestamps = [*timestamps, "_20210112180531", "_20210113130946"]
    #timestamps = [*timestamps, "_20210113163316"]
    #timestamps = ["_20210114174048", "_20210114180643"] #hcp
    #timestamps = [*timestamps, "_20210114175003", "_20210114175808"] #bcp
    #timestamps =[folder[-15:] for folder in os.listdir("./attention_maps") if "random_sample" in folder]
    #timestamps = ["_20210115132243", "_20210115132333"]

    # timestamps = ["_20210216200042", "_20210216142647"]
    # timestamps = ["_20210115145756"]
    # for timestamp in timestamps:
    #     #timestamp = "_20201118183419"
    #     #timestamp = "_20201130160213"
    #     #timestamp = "_20201201103906"
    #     prefix = "./attention_maps/random_sample_{}/".format(timestamp)
    #     df = pd.read_csv(os.path.join(prefix, "cbam_correct.csv"))
    #     #plot_sm_ts(df, dir=prefix+"/cbam_correct/")
    #     plot_slice(df, dir=prefix+"/cbam_correct/")
    #     try:
    #         df = pd.read_csv(os.path.join(prefix, "cbam_wrong.csv"))
    #         #plot_sm_ts(df, dir=prefix+"/cbam_wrong/")
    #         plot_slice(df, dir=prefix + "/cbam_wrong/")
    #     except Exception:
    #         pass
    #     print("timestamp : ", timestamp)

    # df = pd.read_csv("./attention_maps/head motion.csv")
    # dir = "./attention_maps/headmotion"
    # dataset="hcp"
    # for cv in range(1, 6):
    #     print("dataset:{} cv:{}".format(dataset, cv))
    #     grad_cam(dataset, cv, "sm", is_cbam=True, df=df, dir=dir)
    #     grad_cam(dataset, cv, "sm", is_cbam=False, df=df, dir=dir)
    #     grad_cam_ts(dataset, cv, "ts", is_cbam=True, df=df, dir=dir)
    #     grad_cam_ts(dataset, cv, "ts", is_cbam=False, df=df, dir=dir)
    #
    # plot_slice(df, dir=dir)


    # timestamp = "_20201118183419"
    prefix = "./attention_maps/result_figure/"
    df = pd.read_csv(os.path.join("./attention_maps/", "result figure hcp.csv"))
    plot_slice(df, dir=prefix)