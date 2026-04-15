import torch
import torch.nn as nn


def run():
    x = torch.randn(1, 1, requires_grad=True)
    b = 2
    y = x.unsqueeze(0).expand(b, -1, -1)
    loss = y.sum()
    loss.backward()
    print(x.grad)


if __name__ == '__main__':
    run()
