import os
import sys
import shutil
import pickle
import argparse

import numpy as np
import pandas as pd

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../../.."))
from basicts.data.transform import standard_transform


def generate_data(args: argparse.Namespace):
    """Preprocess and generate train/valid/test datasets.

    Args:
        args (argparse): configurations of preprocessing
    """

    target_channel = args.target_channel
    future_seq_len = args.future_seq_len
    history_seq_len = args.history_seq_len
    add_time_of_day = args.tod
    add_day_of_week = args.dow
    add_day_of_month = args.dom
    add_day_of_year = args.doy
    output_dir = args.output_dir
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    data_train_file_path = args.data_train_file_path
    data_val_file_path = args.data_val_file_path
    data_test_file_path = args.data_test_file_path
    graph_file_path = args.graph_file_path
    norm_each_channel = args.norm_each_channel
    if_rescale = not norm_each_channel # if evaluate on rescaled data. see `basicts.runner.base_tsf_runner.BaseTimeSeriesForecastingRunner.build_train_dataset` for details.
    '''
    train/val/test.npz: x,y [B,T_in,N,D] [B,T_out,N,D] 
    data.npz: data [B,N,D]   index.npz: train/val/test [B,3] 
    '''
    # read data
    data_train = np.load(data_train_file_path)
    data_val = np.load(data_val_file_path)
    data_test = np.load(data_test_file_path)
    x_train = data_train['x']
    y_train = data_train['y']
    x_val = data_val['x']
    y_val = data_val['y']
    x_test = data_test['x']
    y_test = data_test['y']

    # 获取data.npz
    def merge_x_y(x_data, y_data):
        merged = np.concatenate((x_data, y_data), axis=1)
        return merged
    train_merged = merge_x_y(x_train, y_train)
    val_merged = merge_x_y(x_val, y_val)
    test_merged = merge_x_y(x_test, y_test)
    all_data = np.concatenate((train_merged, val_merged, test_merged), axis=0) # [B, T, N, D]
    all_data = all_data.reshape(-1, all_data.shape[2], all_data.shape[3])  # (B*T, N, D)
    print("raw time series all_data shape: {0}".format(all_data.shape))

    data = all_data[..., target_channel]
    print("raw time series shape: {0}".format(data.shape))

    # split data
    l, n, f = data.shape
    num_samples = l - (history_seq_len + future_seq_len) + 1
    train_num = round(num_samples * train_ratio)
    valid_num = round(num_samples * valid_ratio)
    test_num = num_samples - train_num - valid_num
    print("number of training samples:{0}".format(train_num))
    print("number of validation samples:{0}".format(valid_num))
    print("number of test samples:{0}".format(test_num))

    num_train = data_train['x'].shape[0]
    num_val = data_val['x'].shape[0]
    num_test = data_test['x'].shape[0]

    time_interval = history_seq_len + future_seq_len
    # 创建索引函数，添加了起始索引
    def create_indices(num_samples, time_interval, start_index=0):
        index_list = []
        for i in range(num_samples):
            base_index = start_index + i * time_interval
            index = (base_index, base_index + time_interval - 1, base_index + time_interval)
            index_list.append(index)
        return index_list
    start_index_val = num_train * time_interval
    start_index_test = start_index_val + num_val * time_interval
    train_index = create_indices(num_train, time_interval)
    valid_index = create_indices(num_val, time_interval, start_index_val)
    test_index = create_indices(num_test, time_interval, start_index_test)

    # normalize data
    scaler = standard_transform
    data_norm = scaler(data, output_dir, train_index, history_seq_len, future_seq_len, norm_each_channel=norm_each_channel)

    # add temporal feature
    feature_list = [data_norm]
    processed_data = np.concatenate(feature_list, axis=-1)

    # save data
    index = {}
    index["train"] = train_index
    index["valid"] = valid_index
    index["test"] = test_index
    with open(output_dir + "/index_in_{0}_out_{1}_rescale_{2}.pkl".format(history_seq_len, future_seq_len, if_rescale), "wb") as f:
        pickle.dump(index, f)

    data = {}
    data["processed_data"] = processed_data
    with open(output_dir + "/data_in_{0}_out_{1}_rescale_{2}.pkl".format(history_seq_len, future_seq_len, if_rescale), "wb") as f:
        pickle.dump(data, f)
    # copy adj
    # shutil.copyfile(graph_file_path, output_dir + "/adj_mx.pkl")
    # adj_mx.npz -> adj_mx.pkl
    adj_mx = np.load(graph_file_path)
    graph = [None, None, adj_mx['adj_mx']]
    with open(output_dir + "/adj_mx.pkl", "wb") as f:
        pickle.dump(graph, f)


if __name__ == "__main__":
    # sliding window size for generating history sequence and target sequence
    HISTORY_SEQ_LEN = 35
    FUTURE_SEQ_LEN = 1

    TRAIN_RATIO = 0.7
    VALID_RATIO = 0.1
    TARGET_CHANNEL = [0,1]                   # target channel(s)

    DATASET_NAME = "NYCTaxi"
    TOD = False                  # if add time_of_day feature
    DOW = False                  # if add day_of_week feature
    DOM = False                  # if add day_of_month feature
    DOY = False                  # if add day_of_year feature

    OUTPUT_DIR = "datasets/" + DATASET_NAME
    DATA_TRAIN_FILE_PATH = "datasets/raw_data/{0}/train.npz".format(DATASET_NAME)
    DATA_VAL_FILE_PATH = "datasets/raw_data/{0}/val.npz".format(DATASET_NAME)
    DATA_TEST_FILE_PATH = "datasets/raw_data/{0}/test.npz".format(DATASET_NAME)
    GRAPH_FILE_PATH = "datasets/raw_data/{0}/adj_mx.npz".format(DATASET_NAME)

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--data_train_file_path", type=str,
                        default=DATA_TRAIN_FILE_PATH, help="Raw traffic readings.")
    parser.add_argument("--data_val_file_path", type=str,
                        default=DATA_VAL_FILE_PATH, help="Raw traffic readings.")
    parser.add_argument("--data_test_file_path", type=str,
                        default=DATA_TEST_FILE_PATH, help="Raw traffic readings.")
    parser.add_argument("--graph_file_path", type=str,
                        default=GRAPH_FILE_PATH, help="Raw traffic readings.")
    parser.add_argument("--history_seq_len", type=int,
                        default=HISTORY_SEQ_LEN, help="Sequence Length.")
    parser.add_argument("--future_seq_len", type=int,
                        default=FUTURE_SEQ_LEN, help="Sequence Length.")
    parser.add_argument("--tod", type=bool, default=TOD,
                        help="Add feature time_of_day.")
    parser.add_argument("--dow", type=bool, default=DOW,
                        help="Add feature day_of_week.")
    parser.add_argument("--dom", type=bool, default=DOM,
                        help="Add feature day_of_week.")
    parser.add_argument("--doy", type=bool, default=DOY,
                        help="Add feature day_of_week.")
    parser.add_argument("--target_channel", type=list,
                        default=TARGET_CHANNEL, help="Selected channels.")
    parser.add_argument("--train_ratio", type=float,
                        default=TRAIN_RATIO, help="Train ratio")
    parser.add_argument("--valid_ratio", type=float,
                        default=VALID_RATIO, help="Validate ratio.")
    parser.add_argument("--norm_each_channel", type=float, help="Validate ratio.")
    args = parser.parse_args()

    # print args
    print("-"*(20+45+5))
    for key, value in sorted(vars(args).items()):
        print("|{0:>20} = {1:<45}|".format(key, str(value)))
    print("-"*(20+45+5))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.norm_each_channel = True
    generate_data(args)
    args.norm_each_channel = False
    generate_data(args)
