import torch
import numpy as np
from matplotlib import pyplot as plt


def read_timeseries(path):

    with open(path) as f:
        seqs = f.readlines()

    return [[(float(t), 0) for t in seq.split(';')[0].split()] for seq in seqs]


def generate_sequence(timeseries, seq_len, log_mode=False):

    ## For the case that Each time_sequence has different length of time-series data.
    TIME_SEQS = []
    EVENT_SEQS = []

    for time_seq in timeseries:

        if not log_mode:
            for idx in range(len(time_seq)-seq_len+1):
                seq = time_seq[idx:idx+seq_len]
                times = [t for (t, e) in seq]
                times = [0] + np.diff(times).tolist()
                events = [e for (t, e) in seq]
                TIME_SEQS.append(times)
                EVENT_SEQS.append(events)

        else:
            for idx in range(len(time_seq) - seq_len + 1):
                seq = time_seq[idx:idx + seq_len]
                times = [t for (t, e) in seq]
                mu = np.mean(times)
                std = np.std(times)
                times = (times-mu)/std
                times = [0] + np.diff(times).tolist()
                events = [e for (t, e) in seq]
                TIME_SEQS.append(times)
                EVENT_SEQS.append(events)

    TIME_SEQS = torch.Tensor(TIME_SEQS)
    EVENT_SEQS = torch.Tensor(EVENT_SEQS)

    return TIME_SEQS, EVENT_SEQS



def plt_lmbda(timeseries, model, seq_len, log_mode=False, dt=0.01, lmbda0=0.2, alpha=0.8, beta=1.0):

    lmbda_dict = dict()
    pred_dict = dict()
    t_span = np.arange(start=timeseries[0][0], stop=timeseries[-1][0]+dt, step=dt)

    # exponential_hwakes : lmbda0, alpha, beta: 0.2, 0.8, 1.0
    # lmbda = lambda0 + alpha*sum(exp{-beta*(t-t_i)})


    lmbda_dict[0] = np.zeros(t_span.shape)


    for t, e in timeseries:
        target = (t_span > t)
        lmbda_dict[0][target] += alpha*np.exp(-beta*(t_span[target]-t))
    lmbda_dict[0] += lmbda0

    # pred_dict[0] = np.zeros(t_span.shape)
    pred_dict[0] = np.zeros(len(timeseries)-seq_len+1)
    test_timeseq, test_eventseq = generate_sequence([timeseries], seq_len, log_mode=log_mode)
    _, _, _, pred_dict[0] = model((test_timeseq, test_eventseq))


    plt.plot(t_span, lmbda_dict[0], color='green')
    plt.plot([t for t, e in timeseries][seq_len-1:], np.array(pred_dict[0].detach()), color='olive')
    plt.scatter([t for t, e in timeseries], [-1 for _ in timeseries], color='blue')
    plt.show()