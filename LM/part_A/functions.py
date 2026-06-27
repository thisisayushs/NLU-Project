import math
import copy
import torch
import torch.nn as nn
from tqdm import tqdm


def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.Linear]:
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias is not None:
                m.bias.data.fill_(0.01)


def train_loop(data, optimizer, criterion, model):
    model.train()
    loss_array = []
    number_of_tokens = []
    for input_ids, labels, n_tokens in data:
        optimizer.zero_grad()
        output = model(input_ids)
        loss = criterion(output.permute(0, 2, 1), labels)
        loss_array.append(loss.item() * n_tokens.item())
        number_of_tokens.append(n_tokens.item())
        loss.backward()
        optimizer.step()
    return sum(loss_array) / sum(number_of_tokens)


def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_array = []
    number_of_tokens = []
    with torch.no_grad():
        for input_ids, labels, n_tokens in data:
            output = model(input_ids)
            loss = eval_criterion(output.permute(0, 2, 1), labels)
            loss_array.append(loss.item() * n_tokens.item())
            number_of_tokens.append(n_tokens.item())
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    ppl = math.exp(loss_to_return)
    return ppl, loss_to_return