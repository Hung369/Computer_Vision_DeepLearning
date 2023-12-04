import torch.nn as nn
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SimpleRNN(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_rnnlayers, n_outputs):
        super(SimpleRNN, self).__init__()
        self.D = n_inputs
        self.M = n_hidden
        self.K = n_outputs
        self.L = n_rnnlayers

        # note: batch_first=True
        # applies the convention that our data will be of shape:
        # (num_samples, sequence_length, num_features)
        # rather than:
        # (sequence_length, num_samples, num_features)
        self.rnn = nn.RNN(
            input_size=self.D,
            hidden_size=self.M,
            num_layers=self.L,
            nonlinearity="relu",
            batch_first=True,
        )
        self.fc = nn.Linear(self.M, self.K)

    def forward(self, X):
      # initial hidden states
      h0 = torch.zeros(self.L, X.size(0), self.M).to(device)

      # get RNN unit output
      # out is of size (N, T, M)
      # 2nd return value is hidden states at each hidden layer
      out, _ = self.rnn(X, h0)

      # we only want h(T) at the final time step
      # N x M -> N x K
      out = self.fc(out[:, -1, :])
      return out
