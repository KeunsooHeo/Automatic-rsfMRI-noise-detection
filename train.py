import time
from model import *
from util import *
from test import test
from copy import deepcopy
import resnet, cbam
from torch.autograd import Variable


def train(config):
    parser = config.parse_args()

    # Define Device
    device = parser.device
    # Data info
    data_keys = parser.data_keys
    dataset = parser.dataset

    ## Train Config
    batch_size = parser.batch_size
    iter_size = parser.iter_size
    lr_step_size = parser.lr_step_size
    iter_log_ratio = parser.iter_log_ratio

    ## Saving config
    model_save_path = parser.path_model
    model_save_overwrite = parser.model_save_overwrite
    model_checkpoint_ratio = parser.model_checkpoint_ratio
    if model_checkpoint_ratio != 0:
        model_checkpoint = int(iter_size * model_checkpoint_ratio)  # 0 for false
    else:
        model_checkpoint = 0
    if not (os.path.isdir(model_save_path)):
        os.makedirs(os.path.join(model_save_path))
    model_base_path = model_save_path + os.path.sep
    load_model = parser.train_load_model
    load_model_name = parser.train_load_model_name
    load_model_name_sm = load_model_name.replace(".pth", "_sm.pth")
    load_model_name_ts = load_model_name.replace(".pth", "_ts.pth")

    test_model = parser.test_model

    ## Hyperparam
    lr = parser.lr
    if parser.mode == "pretrain":
        lr = lr * 0.01

    weight_decay = parser.weight_decay
    momentum = parser.momentum
    gamma = parser.gamma

    # Open train_data
    hdf5 = Hdf5(dataset=dataset, is_train=True, num_source=parser.num_source, data_keys=data_keys, source_dir=parser.path_source)
    train_data_source = hdf5.get_patchlist()
    with open(train_data_source, 'r') as f:
        train_filenames = np.array([line.strip() for line in f.readlines()])

    step_size = len(train_filenames) // batch_size
    step_log = step_size * iter_log_ratio // 1

    path_save_info = parser.path_save_info
    if not (os.path.isdir(path_save_info)):
        os.makedirs(os.path.join(path_save_info))
    path_save_info = path_save_info + os.path.sep + "train_info{}_{}.csv".format(parser.model_save_timestamp,
                                                                                 parser.num_source)
    with open(path_save_info, "w") as f:
        f.write("loss_sm_ts,loss_sm,loss_ts,acc_sm_ts,acc_sm,acc_ts,acc_vote,sen_sm_ts,sen_sm,sen_ts,sen_vote,spec_sm_ts,spec_sm,spec_ts,spec_vote\n")
    with open(path_save_info.replace(".csv", ".txt"), "w") as f:
        f.write("{}".format(parser))
    with open(path_save_info.replace(".csv", "_test.csv"), "w") as f:
        f.write("acc_sm_ts,acc_sm,acc_ts,acc_vote,sen_sm_ts,sen_sm,sen_ts,sen_vote,spec_sm_ts,spec_sm,spec_ts,spec_vote,auc_sm_ts,auc_sm,auc_ts,auc_vote\n")

    ## define & init model
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

    for key in net.keys():
        net[key].apply(init_weights)

    ## train op
    info_gain_weight = torch.tensor(hdf5.infoGain(), dtype=torch.float).to(device)
    loss = nn.CrossEntropyLoss(weight=info_gain_weight)
    #params = list(net["sm_ts"].parameters()) + list(net["sm"].parameters()) + list(net["ts"].parameters())
    opt = parser.optimizer.lower()
    optimizer = {}
    if opt == "sgd":
        for key in net.keys():
            optimizer[key] = torch.optim.SGD(net[key].parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif opt == "adam":
        for key in net.keys():
            optimizer[key] = torch.optim.Adam(net[key].parameters(), lr=lr, weight_decay=weight_decay, betas=[parser.beta1, parser.beta2])
    else:
        raise Exception("unknown optimizer {}, you should choose one of adam or sgd".format(opt))
    lr_scheduler = {}
    for key in net.keys():
        lr_scheduler[key] = torch.optim.lr_scheduler.StepLR(optimizer[key], step_size=lr_step_size, gamma=gamma)

    ## Load Model
    if load_model:
        print("loading model..")
        if parser.mode == "finetune":
            print("mode : fineture")
            net["sm_ts"].load_state_dict(filter_weight(torch.load(load_model_name)), strict=False)
            net["sm"].load_state_dict(filter_weight(torch.load(load_model_name_sm)), strict=False)
            net["ts"].load_state_dict(filter_weight(torch.load(load_model_name_ts)), strict=False)
        else:
            print("loading model : {} complete...".format(parser.train_load_model_name))
            net["sm_ts"].load_state_dict(torch.load(load_model_name), strict=True)
            net["sm"].load_state_dict(torch.load(load_model_name_sm), strict=True)
            net["ts"].load_state_dict(torch.load(load_model_name_ts), strict=True)

    print("device :", device)
    print("training config : ", parser)
    print("training source : ", train_data_source)
    print("training...")
    #for key in net.keys():
    #    optimizer[key].zero_grad()
    for epoch in range(iter_size):
        start_time = time.time()
        d = {key: 0 for key in net.keys()}
        epoch_cost = d.copy()
        step_cost = d.copy()

        d = {key:[] for key in list(net.keys())+["vote"]}
        eval_list = ['acc', 'sen', 'spec']
        #epoch_info = {key: deepcopy(d) for key in eval_list}
        step_info = {key: deepcopy(d) for key in eval_list}
        out_stack = {key: [] for key in ["sm_ts", "sm", "ts", "vote"]}
        label_stack = []

        np.random.shuffle(train_filenames)
        for i in range(step_size):
            # load batch data
            batch_mask = range(i * batch_size, (i + 1) * batch_size)
            train_batch_dic = hdf5.getBatchDicByNames(train_filenames[batch_mask])
            data = torch.tensor(train_batch_dic['data'], dtype=torch.float).to(device)
            if parser.is_fdata:
                _k = 'fdata' if dataset == "bcp" else 'fdata_new'
            else:
                _k = 'tdata'
            tdata = Variable(torch.tensor(train_batch_dic[_k], dtype=torch.float)).to(device)
            label = Variable(torch.tensor(train_batch_dic['label'].squeeze(), dtype=torch.long)).to(device)
            tdata = tdata.view(-1, 1, tdata.shape[-1])

            # forward + backward + optimize
            for key in net.keys():
                optimizer[key].zero_grad()
            out = {}
            out["sm_ts"] = net["sm_ts"](data, tdata)
            out["sm"], _ = net["sm"](data)
            out["ts"], _ = net["ts"](tdata)

            # cost eval & backward
            cost = {}
            for key in net.keys():
                cost[key] = loss(out[key], label)
                cost[key].backward()
                optimizer[key].step()

                epoch_cost[key] += cost[key].item()
                step_cost[key] += cost[key].item()

                out[key] = out[key].data

            ## major voting
            out_arr = np.array([i.cpu().detach().numpy() for i in out.values()])
            vote = major_vote(out_arr)
            out['vote'] = torch.tensor(vote).to(device)

            ## evaluate each model
            label_stack.extend(label.cpu().detach().tolist())
            for key in out.keys():
                out_stack[key].extend(np.argmax(out[key].cpu().detach().tolist(), axis=1))
                step_info['acc'][key] = accuracy(out_stack[key], label_stack)
                step_info['sen'][key] = sensitivity(out_stack[key], label_stack)
                step_info['spec'][key] = specificity(out_stack[key], label_stack)

            if (i + 1) % step_log == 0:
                log = "epoch[{}] step [{:3}/{}] time [{:.1f}s]".format(epoch + 1, i + 1, step_size,
                                                                        time.time() - start_time) + \
                      "|sm_ts loss [{:.5f}] acc [{:.3f}] sen [{:.3f}] spec [{:.3f}]".format(
                          step_cost["sm_ts"] / step_log, *[step_info[key]["sm_ts"] for key in eval_list]) + \
                      "|sm loss [{:.5f}] acc [{:.3f}] sen [{:.3f}] spec [{:.3f}]".format(
                          step_cost["sm"] / step_log, *[step_info[key]["sm"] for key in eval_list]) + \
                      "|ts loss [{:.5f}] acc [{:.3f}] sen [{:.3f}] spec [{:.3f}]".format(
                          step_cost["ts"] / step_log,*[step_info[key]["ts"] for key in eval_list]) + \
                      "|vote acc [{:.5f}] sen [{:.3f}] spec [{:.3f}]".format(
                          *[step_info[key]["vote"] for key in eval_list])
                print(log)

                for key in step_cost.keys():
                    step_cost[key] = 0

        # update lr
        for key in net.keys():
            lr_scheduler[key].step()

        # print epoch info & saving
        log = "[=] epoch [{:}/{:}] time [{:.1f}s]".format(epoch + 1, iter_size, time.time() - start_time) + \
              " |sm_ts loss [{:.5f}] acc [{:.3f}] sen [{:.3f}] spec [{:.3f}]".format(
                  epoch_cost["sm_ts"] / step_size, *[step_info[key]["sm_ts"] for key in eval_list]) + \
              " |sm loss [{:.5f}] acc [{:.3f}] sen [{:.3f}] spec [{:.3f}]".format(
                  epoch_cost["sm"] / step_size, *[step_info[key]["sm"] for key in eval_list]) + \
              " |ts loss [{:.5f}] acc [{:.3f}] sen [{:.3f}] spec [{:.3f}]".format(
                  epoch_cost["ts"] / step_size,*[step_info[key]["ts"] for key in eval_list]) + \
              " |vote acc [{:.5f}] sen [{:.3f}] spec [{:.3f}] ".format(
                  *[step_info[key]["vote"] for key in eval_list])
        print(log)

        with open(path_save_info, "a") as f:
            f.write("{:},{:},{:},{:},{:},{:},{:},{:},{:},{:},{:},{:},{:},{:},{:}\n".format(epoch_cost["sm_ts"] / step_size,
                                                                                       epoch_cost["sm"] / step_size,
                                                                                       epoch_cost["ts"] / step_size,
                                                                                       *[step_info[key1][key2]
                                                                                         for key1 in eval_list for key2 in d.keys()]))

        if model_checkpoint != 0 and (epoch + 1) % model_checkpoint == 0:
            ## saving checkpoint
            print("checkpoint saving..")
            print("save_overwrite :", model_save_overwrite)
            model_name = model_base_path + 'model_{}_{}_{:04}.pth'.format(dataset, parser.num_source, epoch + 1)
            torch.save(net["sm_ts"].state_dict(), model_name)
            torch.save(net["sm"].state_dict(), model_name.replace(".pth", "_sm.pth"))
            torch.save(net["ts"].state_dict(), model_name.replace(".pth", "_ts.pth"))

            ## testing checkpoint
            config.set_defaults(test_model='model_{}_{}_{:04}.pth'.format(dataset, parser.num_source, epoch + 1))
            eval = test(config)
            with open(path_save_info.replace(".csv", "_test.csv"), "a") as f:
                log = ",".join([str(eval[key1][key2]) for key1 in eval.keys() for key2 in d.keys()]) + "\n"
                f.write(log)

    print("training finished!!")

    ## saving model
    print("saving model....")
    model_name = model_base_path + test_model
    torch.save(net["sm_ts"].state_dict(), model_name)
    torch.save(net["sm"].state_dict(), model_name.replace(".pth", "_sm.pth"))
    torch.save(net["ts"].state_dict(), model_name.replace(".pth", "_ts.pth"))
    config.set_defaults(test_model=test_model)

    del net, loss
    torch.cuda.empty_cache()

    print("[!!] Training finished")


if __name__ == "__main__":
    pass