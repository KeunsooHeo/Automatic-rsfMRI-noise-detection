from config import config
from train import *
from test import test
import pandas as pd
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def cross(config):
    parser = config.parse_args()

    # cross validation
    print("cross validation start...")

    df = {}
    for i in range(1, 6):
        print("[!] [{}/5] validation...".format(i))
        test_model = "model_final_{}_{}_{:04}.pth".format(parser.dataset, i, parser.iter_size)
        config.set_defaults(num_source=i, test_model=test_model)
        train(config)
        test_info = test(config)

        print(test_info)
        for key1 in test_info.keys():
            for key2 in test_info[key1].keys():
                df["{}_{}_{}".format(key1, key2, i)] = test_info[key1][key2]

    df = pd.DataFrame.from_dict([df])
    print(df.transpose())
    try:
        df.to_csv("info/acc_cross{}_{}.csv".format(parser.model_save_timestamp, parser.remark))
    except Exception:
        df.to_csv("info/acc_cross{}.csv".format(parser.model_save_timestamp))
    print("[!!] cross validation successfully complete..")
    print("validation id : [{}]".format(parser.model_save_timestamp))

def one(config):
    parser = config.parse_args()
    print("training start..")
    test_model = "model_final_{}_{}_{:04}.pth".format(parser.num_source, parser.dataset, parser.iter_size)
    config.set_defaults(test_model=test_model)
    train(config)
    acc = test(config)
    acc = pd.Series(acc)
    try:
        acc.to_csv("info/acc_one{}_{}.csv".format(parser.model_save_timestamp, parser.remark))
    except Exception:
        acc.to_csv("info/acc_one{}.csv".format(parser.model_save_timestamp))
    print("validation id : [{}]".format(parser.model_save_timestamp))

if __name__ == "__main__":
    config = config()
    parser = config.parse_args()
    print(torch.cuda.device_count())
    if parser.mode == "cross":
        cross(config)
    elif parser.mode == "one":
        one(config)