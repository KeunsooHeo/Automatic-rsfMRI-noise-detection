import argparse
import torch
from datetime import datetime

def config():
    parser = argparse.ArgumentParser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_keys = ["data", "label", "tdata","fdata",'fdata_new']

    model_save_overwrite=False  # debug mode
    timestamp = datetime.today().strftime("_%Y%m%d%H%M%S") if not model_save_overwrite else ""

    parser.add_argument("--remark", default="", type=str, help="just for remarking, it does not effect on program")

    # pytorch base
    parser.add_argument("--device", default=device)

    # data
    parser.add_argument("--dataset", default="hcp", type=str, help="bcp, hcp, mb6, std")
    parser.add_argument("--data_keys", default=data_keys, type=list, help="")
    parser.add_argument("--is_fdata", default=False, type=bool, help="choose 1d data as frequency data (if False use temporal data)")

    # training mode
    parser.add_argument("--mode", default="cross", type=str, help="one, cross, finetune, pretrain")
    parser.add_argument("--init" , default="he", type=str, help="[xavier, he, default]")

    # model
    parser.add_argument("--network", default="cbam_test3_234", type=str, help="network, one of [resnet_test2, cbam_test3_234]")

    # training hyperparams
    parser.add_argument("--lr", default=1e-2, type=float, help="learning rate")
    parser.add_argument("--batch_size", default=30, type=int, help="batch size")
    parser.add_argument("--iter_size", default=300, type=int, help="maximum of iterations")
    parser.add_argument("--pretrain_iter_size", default=300, type=int, help="maximum of iterations")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="weight decay for regularization ")
    parser.add_argument("--momentum", default=0.9, type=float, help="optimizer momentum")
    parser.add_argument("--optimizer", default="sgd", type=str, help="select optimizer sgd or adam")
    parser.add_argument("--beta1", default=0.9, type=float, help="adam optimizer beta1")
    parser.add_argument("--beta2", default=0.999, type=float, help="adam optimizer beta2")

    # printing training log
    parser.add_argument("--lr_step_size", default=100, type=int, help="step per lr learning")
    parser.add_argument("--iter_log_ratio", default=0.20, type=float, help="ratio to print log step by step")
    parser.add_argument("--gamma", default=0.1, type=float, help="gamma for lr learning")

    # source file
    parser.add_argument("--num_source", default=3, type=int, help="number of using source")
    parser.add_argument("--path_source", default="patchList_new", type=str, help="path to save model")

    # save & load
    parser.add_argument("--model_save_overwrite", default=model_save_overwrite, type=bool, help="Debug mode")
    parser.add_argument("--model_save_timestamp", default=timestamp, type=str, help="")
    parser.add_argument("--path_model", default="model/model{}".format(timestamp), type=str, help="path to save and to load model")
    parser.add_argument("--model_checkpoint_ratio", default=0.20, type=float, help="0 for False")
    parser.add_argument("--test_model", default="model.pth", type=str, help="")
    parser.add_argument("--path_save_info", default="info/info{}".format(timestamp), type=str, help="")

    # pre-train
    parser.add_argument("--train_load_model", default=False, type=bool, help="true or false to load model during training")
    parser.add_argument("--train_load_model_name", default="model.pth", type=str, help="")

    return parser

if __name__ == "__main__":
    config = config()
    p = config.parse_args()
    print(p)