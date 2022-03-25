import time
from model import *
from util import *
import cbam
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

class Trainer:
    def __init__(self, config):
        self.config = config
        # self.parser = config.parse_args()
        # Define Device
        self.device = parser.device  # parser.device
        self.data_keys = parser.data_keys
        self.dataset = parser.dataset

    def train(self):
        config = self.config
        parser = config.parse_args()
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

        test_model = parser.test_model

        ## Hyperparam
        lr = parser.lr
        weight_decay = parser.weight_decay
        momentum = parser.momentum
        gamma = parser.gamma

        # Open train_data
        dataloader = Hdf5(dataset=self.dataset, is_train=True, num_source=parser.num_source,
                                  data_keys=self.data_keys, source_dir=parser.path_source)

        # _, _label = dataloader.load_data()

        train_data_source = dataloader.get_patchlist()
        with open(train_data_source, 'r') as f:
            train_filenames = np.array([line.strip() for line in f.readlines()])

        size = len(train_filenames)
        step_size = size // batch_size
        step_log = step_size * iter_log_ratio // 1
        idx = np.array(range(size))

        path_save_info = parser.path_save_info
        if not (os.path.isdir(path_save_info)):
            os.makedirs(os.path.join(path_save_info))
        path_save_info = path_save_info + os.path.sep + "train_info{}_{}.csv".format(parser.model_save_timestamp,
                                                                                     parser.num_source)
        with open(path_save_info, "w") as f:
            f.write("loss,acc,sen,spec\n")
        with open(path_save_info.replace(".csv", ".txt"), "w") as f:
            f.write("{}".format(parser))
        with open(path_save_info.replace(".csv", "_test.csv"), "w") as f:
            f.write("acc,sen,spec,auc\n")
        network = parser.network.lower()
        if "resnet" in network or "cbam" in network:
            if network == "cbam_test3_234":
                res = cbam.resnet_test3_234
                print("network : cbam_test3_234")
            else:
                raise ValueError("No resnet match {}".format(network))

        ## define & init models
        self.net = nn.DataParallel(Resnet_fusion(res=res)).to(self.device)

        ## train op
        info_gain_weight = torch.tensor(dataloader.infoGain(), dtype=torch.float).to(self.device)
        loss = nn.CrossEntropyLoss(weight=info_gain_weight)
        opt = parser.optimizer.lower()
        if opt == "sgd":
            optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif opt == "adam":
            optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay,
                                         betas=(parser.beta1, parser.beta2))
        else:
            raise Exception("unknown optimizer {}, you should choose one of adam or sgd".format(opt))
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=gamma)

        ## Load Model
        if load_model:
            print("loading model : {} complete...".format(parser.train_load_model_name))
            self.net.load_state_dict(torch.load(load_model_name), strict=True)

        print("self.device :", self.device)
        print("training config : ", parser)
        print("training...")
        for epoch in range(iter_size):
            start_time = time.time()
            epoch_cost = 0
            step_cost = 0

            eval_list = ['acc', 'sen', 'spec']
            step_info = {key: 0 for key in eval_list}
            out_stack = []
            label_stack = []
            self.net.train()

            np.random.shuffle(idx)
            for i in range(step_size):
                # load batch data
                batch_mask = range(i * batch_size, (i + 1) * batch_size)
                train_batch_dic = dataloader.getBatchDicByNames(train_filenames[idx[batch_mask]])
                data = torch.tensor(train_batch_dic['data'], dtype=torch.float).to(self.device)
                tdata = torch.tensor(train_batch_dic['tdata'], dtype=torch.float).to(self.device)
                label = torch.tensor(train_batch_dic['label'].squeeze(), dtype=torch.long).to(self.device)
                tdata = tdata.view(-1, 1, tdata.shape[-1])

                # forward + backward + optimize
                optimizer.zero_grad()
                out = self.net(data, tdata)
                cost = loss(out, label)
                cost.backward()

                epoch_cost += cost.item()
                step_cost += cost.item()

                out = out.data
                optimizer.step()

                ## evaluate each model
                label_stack.extend(label.cpu().detach().tolist())
                out_stack.extend(np.argmax(out.cpu().detach().tolist(), axis=1))
                step_info['acc'] = accuracy(out_stack, label_stack)
                step_info['sen'] = sensitivity(out_stack, label_stack)
                step_info['spec'] = specificity(out_stack, label_stack)

                if (i + 1) % step_log == 0:
                    log = "epoch[{}] step [{:3}/{}] time [{:.1f}s]".format(epoch + 1, i + 1, step_size,
                                                                           time.time() - start_time) + \
                          "| loss [{:.5f}] acc [{:.3f}] sen [{:.3f}] spec [{:.3f}]".format(
                              step_cost / step_log, *[step_info[key] for key in eval_list])
                    print(log)

                    step_cost = 0

            # update lr
            lr_scheduler.step()

            # print epoch info & saving
            log = "[=] epoch [{:}/{:}] time [{:.1f}s]".format(epoch + 1, iter_size, time.time() - start_time) + \
                  " | loss [{:.5f}] acc [{:.3f}] sen [{:.3f}] spec [{:.3f}]".format(
                      epoch_cost / step_size, *[step_info[key] for key in eval_list])
            print(log)

            with open(path_save_info, "a") as f:
                f.write("{:},{:},{:},{:}\n".format(epoch_cost / step_size, *[step_info[key1] for key1 in eval_list]))

            if model_checkpoint != 0 and (epoch + 1) % model_checkpoint == 0:
                ## saving checkpoint
                print("checkpoint saving..")
                print("save_overwrite :", model_save_overwrite)
                model_name = model_base_path + 'model_{}_{}_{:04}.pth'.format(self.dataset, parser.num_source,
                                                                              epoch + 1)
                torch.save(self.net.state_dict(), model_name)

                ## testing checkpoint
                eval = self.test()
                with open(path_save_info.replace(".csv", "_test.csv"), "a") as f:
                    log = ",".join([str(eval[key1]) for key1 in eval.keys()]) + "\n"
                    f.write(log)

        print("training finished!!")

        ## saving model
        print("saving model....")
        model_name = model_base_path + test_model
        torch.save(self.net.state_dict(), model_name)
        config.set_defaults(test_model=test_model)

        torch.cuda.empty_cache()

        print("[!!] Training finished")

    def test(self):
        parser = config.parse_args()
        dataloader = Hdf5(dataset=self.dataset, is_train=False, num_source=parser.num_source,
                                  data_keys=self.data_keys, source_dir=parser.path_source)

        # load model
        self.net.eval()

        # print(net)

        # _, _label = dataloader.load_data()
        source = dataloader.get_patchlist()
        with open(source, 'r') as f:
            test_filenames = np.array([line.strip() for line in f.readlines()])
            # np.random.shuffle(test_filenames)

        # testing
        print("testing config : ", parser)
        print("testing...")
        eval_list = ['acc', 'sen', 'spec', 'auc']
        step_info = {key2: 0 for key2 in eval_list}

        step_size = len(test_filenames)
        step_log = int(step_size * parser.iter_log_ratio)
        start_time = time.time()

        out_stack = []
        label_stack = []
        with torch.no_grad():
            for step in range(step_size):
                # load batch data
                train_batch_dic = dataloader.getDataDicByName(test_filenames[step])
                data = torch.tensor(train_batch_dic['data'], dtype=torch.float).to(self.device)
                tdata = torch.tensor(train_batch_dic['tdata'], dtype=torch.float).to(self.device)
                tdata = tdata.view(1, 1, -1)
                label = torch.tensor(train_batch_dic['label'].squeeze(), dtype=torch.long).to(self.device)

                # forward
                out = self.net(data,tdata)

                label_stack.append(label.cpu().detach().tolist())
                out_stack.extend(np.argmax(out.cpu().detach().tolist(), axis=1))
                step_info['acc'] = accuracy(out_stack, label_stack)
                step_info['sen'] = sensitivity(out_stack, label_stack)
                step_info['spec'] = specificity(out_stack, label_stack)

                if step_log != 0 and (step + 1) % step_log == 0:
                    log = "step [{:3}/{}] time [{:.1f}s]".format(step + 1, step_size, time.time() - start_time) + \
                          " | acc [{:.3f}] sen [{:.3f}] spec [{:.3f}]".format(
                              *[step_info[key] for key in eval_list[:-1]])
                    print(log)

        out = {key: 0. for key in eval_list}
        print("{}{},{},{},{},{}".format(parser.dataset, parser.model_save_timestamp, *eval_list))
        out['acc'] = accuracy(out_stack, label_stack)
        out['sen'] = sensitivity(out_stack, label_stack)
        out['spec'] = specificity(out_stack, label_stack)
        out['auc'] = roc_auc_score(y_score=out_stack, y_true=label_stack)
        print("{},{},{},{}".format(*[out[key1] * 100 for key1 in eval_list]))
        print("[!!] Testing complete info save test/info{}_{}.csv".format(parser.model_save_timestamp,
                                                                          parser.num_source))

        return out

    def one(self):
        parser = config.parse_args()
        self.train()
        acc = self.test()
        acc = pd.Series(acc)
        try:
            acc.to_csv("info/acc_one{}_{}.csv".format(parser.model_save_timestamp, parser.remark))
        except Exception:
            acc.to_csv("info/acc_one{}.csv".format(parser.model_save_timestamp))
        print("validation id : [{}]".format(parser.model_save_timestamp))

    def cross(self):
        parser = config.parse_args()
        df = {}
        for i in range(1, 6):
            print("[!] [{}/5] validation...".format(i))
            test_model = "model_final_{}_{}_{:04}.pth".format(parser.dataset, i, parser.iter_size)
            config.set_defaults(num_source=i, test_model=test_model)
            self.train()
            test_info = self.test()

            print(test_info)
            for key1 in test_info.keys():
                if key1 not in df.keys():
                    df[key1] = test_info[key1]
                else:
                    df[key1] += test_info[key1]

        for key1 in df.keys():
            df[key1] = [df[key1] / 5]

        df = pd.DataFrame.from_dict(df)
        print(self.dataset, parser.model_save_timestamp)
        print(df)
        try:
            df.to_csv("info/acc_cross{}_{}.csv".format(parser.model_save_timestamp, parser.remark))
        except Exception:
            df.to_csv("info/acc_cross{}.csv".format(parser.model_save_timestamp))
        print("[!!] cross validation successfully complete..")
        print("validation id : [{}]".format(parser.model_save_timestamp))

import matplotlib.pyplot as plt
if __name__ == "__main__":
    config = config()
    parser = config.parse_args()
    trainer = Trainer(config)
    trainer.cross()