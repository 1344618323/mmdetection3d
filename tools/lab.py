import torch


def run():
    x = torch.randn(1, 2, 2, 2)
    print(x)
    s = x.cumsum(1, dtype=torch.float32)
    print(s)


if __name__ == '__main__':
    run()
