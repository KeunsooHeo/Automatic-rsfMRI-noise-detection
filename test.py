import os, time
import numpy as np
import torch
import torch.nn as nn
from model import *
from util import *
import resnet, cbam
import pandas as pd

def test(config):
    parser = config.parse_args()

    # Define Device
    device = parser.device
    # Data info
    data_keys = parser.data_keys
    # Test config
    iter_log_ratio = parser.iter_log_ratio

    # load test data
    dataset = parser.dataset
    hdf5 = Hdf5(dataset=dataset, is_train=False, num_source=parser.num_source, data_keys=data_keys, source_dir=parser.path_source)
    source = hdf5.get_patchlist()
    with open(source, 'r') as f:
        test_filenames = np.array([line.strip() for line in f.readlines()])
        np.random.shuffle(test_filenames)

    # load model
    network = parser.network.lower()
    if "resnet" in network or "cbam" in network:
        if network == "resnet_test2":
            res = resnet.resnet_test2
            print("network : resnet_test2")
        elif network == "cbam_test3_234":
            res = cbam.resnet_test3_234
            print("network : cbam_test3_234")
        else:
            raise ValueError("No resnet match {}".format(network))

    net = {}
    net["sm_ts"] = nn.DataParallel(Resnet_sm_ts(res=res)).to(device)
    net["sm"] = nn.DataParallel(Resnet_sm(res=res)).to(device)
    net["ts"] = nn.DataParallel(Resnet_ts(res=res)).to(device)

    #print(net)
    net["sm_ts"].load_state_dict(torch.load(parser.path_model + os.path.sep + parser.test_model))
    net["sm"].load_state_dict(torch.load(parser.path_model + os.path.sep + parser.test_model.replace(".pth", "_sm.pth")))
    net["ts"].load_state_dict(torch.load(parser.path_model + os.path.sep + parser.test_model.replace(".pth", "_ts.pth")))

    def set_bn_eval(m):
        if isinstance(m, (nn.BatchNorm3d, nn.BatchNorm1d)):
            m.eval()

    for k in net.keys():
        for name, param in net[k].named_parameters():
            param.requires_grad = False
        net[k].apply(set_bn_eval)

    # testing
    print("device :", device)
    print("testing config : ", parser)
    print("testing source : ", source)
    print("testing...")
    d = ["sm_ts","sm","ts","vote"]
    eval_list = ['acc', 'sen', 'spec']
    step_info = {key2: {key:[] for key in ["sm_ts","sm","ts","vote"]} for key2 in eval_list}

    step_size = len(test_filenames) #// batch_size + 1
    step_log = int(step_size * iter_log_ratio)
    start_time = time.time()

    out_stack ={key:[] for key in ["sm_ts","sm","ts","vote"]}
    label_stack = []
    with torch.no_grad():
        for step in range(step_size):
            # load batch data
            train_batch_dic = hdf5.getDataDicByName(test_filenames[step])
            data = torch.tensor(train_batch_dic['data'], dtype=torch.float).to(device)
            if parser.is_fdata:
                _k = 'fdata' if dataset == "bcp" else 'fdata_new'
            else:
                _k = 'tdata'
            tdata = torch.tensor(train_batch_dic[_k], dtype=torch.float).to(device)
            label = torch.tensor(train_batch_dic['label'].squeeze(), dtype=torch.long).to(device)
            tdata = tdata.view(-1, 1, tdata.shape[-1])

            # forward
            out = {}
            out["sm_ts"] = net["sm_ts"](data, tdata)
            out["sm"], _ = net["sm"](data)
            out["ts"], _ = net["ts"](tdata)

            # loss
            for key in net.keys():
                out[key] = out[key].data

            ## major voting
            out_arr = np.array([i.cpu().detach().numpy() for i in out.values()])
            vote = major_vote(out_arr)
            out["vote"] = torch.tensor(vote).to(device)

            label_stack.append(label.cpu().detach().tolist())
            for key in out.keys():
                out_stack[key].extend(np.argmax(out[key].cpu().detach().tolist(), axis=1))
                step_info['acc'][key] = accuracy(out_stack[key], label_stack)
                step_info['sen'][key] = sensitivity(out_stack[key], label_stack)
                step_info['spec'][key] = specificity(out_stack[key], label_stack)


            if step_log !=0 and (step + 1) % step_log == 0:
                log = "step [{:3}/{}] time [{:.1f}s]".format(step + 1, step_size, time.time() - start_time) + \
                " | sm_ts acc [{:.3f}] sen [{:.3f}] spec [{:.3f}]".format(*[step_info[key]["sm_ts"] for key in eval_list]) + \
                " | sm acc [{:.3f}] sen [{:.3f}] spec [{:.3f}]".format(*[step_info[key]["sm"] for key in eval_list]) + \
                " | ts acc [{:.3f}] sen [{:.3f}] spec [{:.3f}]".format(*[step_info[key]["ts"] for key in eval_list]) + \
                " | vote acc [{:.3f}] sen [{:.3f}] spec [{:.3f}]".format(*[step_info[key]["vote"] for key in eval_list])
                print(log)

    out = {key:{key:0. for key in ["sm_ts","sm","ts","vote"]} for key in eval_list}
    print("{}{},{},{},{},{}".format(parser.dataset,parser.model_save_timestamp,*eval_list))
    for key2 in d:
        out['acc'][key2] = accuracy(out_stack[key2], label_stack)
        out['sen'][key2] = sensitivity(out_stack[key2], label_stack)
        out['spec'][key2] = specificity(out_stack[key2], label_stack)
        print("{},{},{},{},{}".format(key2,*[out[key1][key2]* 100 for key1 in eval_list]))
    print("[!!] Testing complete info save test/info{}_{}.csv".format(parser.model_save_timestamp, parser.num_source))

    return out

if __name__ == "__main__":
    from config import config
    from grad_cam import get_timestamp
    config = config()
    dataset="bcp_unseen"
    timestamp = get_timestamp(dataset="bcp", is_cbam=False)

    parser = config.parse_args()

    # cross validation
    print("cross validation start...")

    df = {}
    for i in range(1, 6):
        print("[!] [{}/5] validation...".format(i))
        test_model = "model_final_{}_{}_{:04}.pth".format("bcp", i, parser.iter_size)
        config.set_defaults(timestamp=timestamp, dataset=dataset, num_source=i, test_model=test_model, network="resnet_test2", path_model="model/model{}".format(timestamp))
        test_info = test(config)

        print(test_info)
        for key1 in test_info.keys():
            for key2 in test_info[key1].keys():
                df["{}_{}_{}".format(key1, key2, i)] = test_info[key1][key2]

    # print acc
    # for e in ["acc", "sen", "spec"]:
    #    df["{}_avg".format(e)] = df[["{}_{}".format(e, a) for a in ['sm_ts', 'sm', 'ts', 'vote']]].mean(axis=1)
    # df = df.append(df.mean(axis=0), ignore_index=True)

    # df.index = ['sm_ts', 'sm', 'ts', 'vote', 'avg']
    df = pd.DataFrame.from_dict([df])
    print(df.transpose())
    try:
        df.to_csv("info/acc_cross{}_{}.csv".format(parser.model_save_timestamp, parser.remark))
    except Exception:
        df.to_csv("info/acc_cross{}.csv".format(parser.model_save_timestamp))
    print("[!!] cross validation successfully complete..")
    print("validation id : [{}]".format(parser.model_save_timestamp))