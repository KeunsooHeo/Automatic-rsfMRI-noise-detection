import nibabel as nib
import h5py
import os
import numpy as np
from grad_cam import remove_pad

def min_max_norm(array):
    max = np.max(array)
    min = np.min(array)
    max_range = np.max([max, -min])

    return array / max_range

nii_data_path = "input nii folder path"
sample_nums = os.listdir(nii_data_path)

hdf_data_path = "output hdf folder path"
if not os.path.exists(hdf_data_path):
    os.mkdir(hdf_data_path)

patch_list_name = "list.txt"
with open(os.path.join("./patchList_new", patch_list_name), "w") as f:
    pass
# sample
for sample_num in sample_nums:
    nii = nib.load(os.path.join(nii_data_path + sample_num, "melodic_IC.nii.gz"))
    nii2array = np.asarray(nii.dataobj)
    with open(os.path.join(nii_data_path + sample_num, "melodic_mix"), "r") as f:
        ts = np.array([[float(i) for i in line.split()] for line in f.readlines()])
    with open(os.path.join(nii_data_path + sample_num, "hand_labels_noise.txt"), "r") as f:
        noise_label = f.readline().rstrip()
        noise_label = eval(noise_label)

    length = nii2array.shape[-1]
    for i in range(length): # comp
        s_IC = nii2array[:,:,:,i]
        t_IC = ts[:,i]

        s_IC = min_max_norm(s_IC)
        s_IC = remove_pad(s_IC, "unseen")
        t_IC = min_max_norm(t_IC)
        s_IC = np.expand_dims(s_IC, axis=(0, 1))
        t_IC = np.expand_dims(t_IC, axis=(0, 1))
        f_IC = np.zeros((1,1,1)) # dummy f_data
        if not os.path.exists(hdf_data_path + sample_num):
            os.mkdir(hdf_data_path + sample_num)
        label = int(not (i+1) in noise_label)
        dic = {"data":s_IC, "tdata":t_IC, "fdata":f_IC, "label":label}
        with h5py.File(os.path.join(hdf_data_path + sample_num, "Comp{:03}.hdf5".format(i+1)), 'w') as f:
            f.create_dataset("data", data=s_IC)
            f.create_dataset("tdata", data=t_IC)
            f.create_dataset("fdata", data=f_IC)
            f.create_dataset("label", data=label)

        with h5py.File(os.path.join(hdf_data_path + sample_num, "Comp{:03}.hdf5".format(i + 1)), 'r') as f:
            a= np.array(f["data"])
            pass

        with open(os.path.join("./patchList_new", patch_list_name), "a") as f:
            f.write(os.path.join(hdf_data_path + sample_num, "Comp{:03}.hdf5\n".format(i + 1)))

        print("[{}] Comp[{}/{}]".format(sample_num,i+1,length))
