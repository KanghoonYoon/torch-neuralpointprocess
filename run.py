

import numpy as np
import tqdm
import torch

from argparse import ArgumentParser

from torch.utils.data import DataLoader

from utils import read_timeseries, generate_sequence, plt_lmbda
from module import GTPP



if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default='exponential_hawkes')
    parser.add_argument("--model", type=str, default='GTPP')
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--emb_dim", type=int, default=10)
    parser.add_argument("--hid_dim", type=int, default=64)
    parser.add_argument("--mlp_layer", type=int, default=2)
    parser.add_argument("--mlp_dim", type=int, default=64)
    parser.add_argument("--event_class", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--prt_evry", type=int, default=5)
    parser.add_argument("--early_stop", type=bool, default=True)
    ## Alpha ??
    parser.add_argument("--alpha", type=float, default=0.05)

    parser.add_argument("--importance_weight", action="store_true")
    parser.add_argument("--log_mode", type=bool, default=False)


    config = parser.parse_args()

    path = 'data/'

    if config.data == 'exponential_hawkes':

        train_data = read_timeseries(path + config.data + '_training.csv')
        val_data = read_timeseries(path + config.data + '_validation.csv')
        test_data = read_timeseries(path + config.data + '_testing.csv')

    train_timeseq, train_eventseq = generate_sequence(train_data, config.seq_len, log_mode=config.log_mode)
    train_loader = DataLoader(torch.utils.data.TensorDataset(train_timeseq, train_eventseq), shuffle=True, batch_size=config.batch_size)
    val_timeseq, val_eventseq = generate_sequence(val_data, config.seq_len, log_mode=config.log_mode)
    val_loader = DataLoader(torch.utils.data.TensorDataset(val_timeseq, val_eventseq), shuffle=False, batch_size=len(val_data))

    model = GTPP(config)

    best_loss = 1e3
    patients = 0
    tol = 20

    for epoch in range(config.epochs):

        model.train()

        loss1 = loss2 = loss3 = 0

        for batch in train_loader:
            loss, log_lmbda, int_lmbda, lmbda = model.train_batch(batch)

            loss1 += loss
            loss2 += log_lmbda
            loss3 += int_lmbda


        model.eval()

        for batch in val_loader:
            val_loss, val_log_lmbda, val_int_lmbda, _ = model(batch)

        if best_loss > val_loss:
            best_loss = val_loss.item()
        else:
            patients += 1
            if patients >= tol:
                print("Early Stop")
                print("epoch", epoch)
                plt_lmbda(train_data[0], model=model, seq_len=config.seq_len, log_mode=config.log_mode)
                break

        if epoch % config.prt_evry == 0:
            print("Epochs:{}".format(epoch))
            print("Training Negative Log Likelihood:{}   Log Lambda:{}:   Integral Lambda:{}".format(loss1/config.batch_size, -loss2 / config.batch_size, loss3 / config.batch_size))
            print("Validation Negative Log Likelihood:{}   Log Lambda:{}:   Integral Lambda:{}".format(val_loss / config.batch_size,
                                                                                            -val_log_lmbda / config.batch_size,
                                                                                            val_int_lmbda/ config.batch_size))
            plt_lmbda(train_data[0], model=model, seq_len=config.seq_len, log_mode=config.log_mode)
            # plt_lmbda(test_data[0], model=model, seq_len=config.seq_len, log_mode=config.log_mode)


    print("end")
















