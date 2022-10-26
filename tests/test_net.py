from pathlib import Path
from models.test_net import Net
from torch import nn, rand

def test_net():
    net = Net()
    img = rand((1, 3, 224, 224), requires_grad=True)
    out = net(img)
    print(out.shape)
    return


if __name__ == "__main__":
    test_net()