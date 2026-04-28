import torch
import torch.nn as nn
import torch.optim as optim


def run():
    x = torch.randn(1, 2, 3)
    index = torch.tensor(
        [
            [0, 0, 1]
        ]
    )
    index = index[:, None, :].expand(-1, x.shape[1], -1)
    print(x)
    print(index)
    print(x.gather(dim=-1, index=index))


if __name__ == '__main__':
    run()
