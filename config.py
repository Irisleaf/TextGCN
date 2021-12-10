import argparse
import torch


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='./data/', type=str)
    parser.add_argument("--graph_dir", default='data/graph/', type=str)
    parser.add_argument("--saved_model_path", default='./saved_model/', type=str)

    parser.add_argument("--epoch", default=100, type=int)
    # 100
    # parser.add_argument("--device", default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    parser.add_argument("--device", default='cpu')
    parser.add_argument("--lr", default=0.01, type=float)
    #0.01
    parser.add_argument("--hidden_size", default=2048, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--val_ratio", default=0.2, type=float)
    #0.1
    parser.add_argument("--seed", default=2021, type=int)


    return parser.parse_args()