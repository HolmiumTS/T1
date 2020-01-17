import torch
import torch.nn.functional as nnf
from torch.autograd import Variable
from torch import nn
from torch.utils import data
from DataSet import MyDataset

# Hyper params
EPOCH = 3
INPUT_SIZE = 133
OUTPUT_SIZE = 64
HIDDEN_SIZE = 128
TIME_STEP = 49
NUM_LAYERS = 1
BATCH_FIRST = True
BATCH_SIZE = 16
LR = 0.07

print("##")
train_data = MyDataset()
print("##")

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
        )

        self.out = nn.Linear(HIDDEN_SIZE, 310)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)  # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


if __name__ == '__main__':
    rnn = RNN()
    print(rnn)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

    # training and testing
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data
            b_x = torch.tensor(b_x, dtype=torch.float32)
            output = rnn(b_x)  # rnn output
            loss = loss_func(output, b_y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

    pred = []
    for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data
        b_x = torch.tensor(b_x, dtype=torch.float32)
        test_output = rnn(b_x)  # (samples, time_step, input_size)
        pred_y = torch.max(test_output, 1)[1].data.numpy()
        pred.extend(pred_y)
        # accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
        # print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
    cnt = 0.0
    for i in range(len(pred)):
        if pred[i] == train_data.train_label[i]:
            cnt += 1
    print(cnt / float(len(pred)))
