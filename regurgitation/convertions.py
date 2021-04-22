import torch
# from config import Config

def tensor_prob_to_one_hot(tensor_prob, dim=1, cuda=True):
    max_idx = torch.argmax(tensor_prob, dim=dim, keepdim=True)
    one_hot = torch.FloatTensor(tensor_prob.size())
    if cuda:
        one_hot = one_hot.cuda()
    one_hot.zero_()
    one_hot.scatter_(1, max_idx, 1)
    return one_hot

def tensor_label_to_one_hot(label, class_num, cuda=True):
    one_hot = torch.zeros(label.size(0), class_num)
    max_idx = label[:,None].long()
    if cuda:
        one_hot = one_hot.cuda()
    one_hot.scatter_(1, max_idx, 1)
    return one_hot


def label_to_one_hot(label, class_num, cuda=True):
    one_hot = torch.FloatTensor(label.shape[0], class_num, label.shape[1], label.shape[2])
    max_idx = torch.unsqueeze(label, dim=1)
    if cuda:
        one_hot = one_hot.cuda()
    one_hot.zero_()
    one_hot.scatter_(1, max_idx, 1)
    return one_hot


def one_hot_to_label(one_hot, dim=1):
    max_idx = torch.argmax(one_hot, dim=dim, keepdim=False).long()
    return max_idx


def prob_to_one_hot(img, dim=1, cuda=True):
    max_idx = torch.argmax(img, dim=dim, keepdim=True)
    one_hot = torch.FloatTensor(img.shape)
    if cuda:
        one_hot = one_hot.cuda()
    one_hot.zero_()
    one_hot.scatter_(1, max_idx, 1)
    return one_hot


def prob_to_label(img, dim=1):
    max_idx = torch.argmax(img, dim=dim, keepdim=False)
    return max_idx


def binary_prob_to_label(img, threshold=0.5):
    return torch.squeeze((img > threshold).long())


if __name__ == '__main__':
    a = torch.randn(2, 3, 128, 128)
    b = prob_to_label(a)
    print(b)
    print('----------')
    a = torch.randn(2, 3, 128, 128)
    b = prob_to_one_hot(a)
    print(b)
    print('----------')
    a = torch.zeros(1, 128, 128).long()
    for i in range(128):
        for j in range(128):
            a[0, i, j] = 2
    b = label_to_one_hot(a, class_num=4, cuda=False)
    print(b)
    print('----------')
    a = torch.rand(1, 128, 128)
    b = binary_prob_to_label(a)
    print(b)
    print('----------')
    x = torch.zeros(1, 3, 128, 128).long()
    for i in range(128):
        for j in range(128):
            x[0, 1, i, j] = 1
    y = one_hot_to_label(x)
    print('----------')
