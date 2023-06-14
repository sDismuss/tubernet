import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="data\\")
    parser.add_argument('--width', type=int, default=224)
    parser.add_argument('--height', type=str, default=224)

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batches', type=int, default=32)
    parser.add_argument('--seed', type=int, default=32)

    args = parser.parse_args()
    return args
