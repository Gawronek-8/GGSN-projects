from torch import nn


class _SimpleBlock(nn.Module):
    def __init__(self, neurons_in, neurons_out, activation: nn.Module, dropout, is_batch_norm, is_skipping_conn):
        super().__init__()

        self.linear1 = nn.Linear(neurons_in, neurons_out)
        self.linear2 = nn.Linear(neurons_out, neurons_out)

        if isinstance(activation, type):
            self.activation1 = activation()
            self.activation2 = activation()
        elif isinstance(activation, nn.Module):
            self.activation1 = activation
            self.activation2 = activation
        else:
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()

        if is_batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(neurons_out)
            self.batch_norm2 = nn.BatchNorm1d(neurons_out)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

        if neurons_in == neurons_out and is_skipping_conn:
            self.skip_connection = nn.Identity()
        elif neurons_in != neurons_out and is_skipping_conn:
            self.skip_connection = nn.Linear(neurons_in, neurons_out)


    def forward(self, x):

        skip_val = x

        out = self.linear1(x)
        if hasattr(self, 'batch_norm1'):
            out = self.batch_norm1(out)

        out = self.activation1(out)

        if hasattr(self, 'dropout'):
            out = self.dropout(out)

        out = self.linear2(out)
        if hasattr(self, 'batch_norm2'):
            out = self.batch_norm2(out)

        if hasattr(self, 'skip_connection'):
            out = self.activation2(out + self.skip_connection(skip_val))
        else:
            out = self.activation2(out)

        if hasattr(self, 'dropout'):
            out = self.dropout(out)

        return out



class Model(nn.Module):

    def __init__(self, neurons : int, dropouts, is_batchnorm, activations, is_skipping_conn, input_shape, output_shape, name = "model_default_name"):
        super().__init__()

        self.train_info = None
        self.first_layer = _SimpleBlock(input_shape, neurons, activations[0], dropouts[0], is_batchnorm[0], is_skipping_conn[0])
        self.hidden = nn.Sequential(
            *[_SimpleBlock(neurons // i , neurons // (i + 1), activations[i], dropouts[i], is_batchnorm[i], is_skipping_conn[i])
              for i in range(1, len(activations))],
        )
        self.output_layer = nn.Linear(neurons // max(len(activations), 1), output_shape)
        self.name = name
        self.model_info = {"model": {
            "neurons": neurons,
            "dropouts": dropouts,
            "is_batchnorm": is_batchnorm,
            "activations": [act.__name__ if isinstance(act, type) else act.__class__.__name__ for act in activations ],
            "is_skipping_conn": is_skipping_conn,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "name": self.name
        }}

    def forward(self, x):

        out = self.first_layer(x)
        out = self.hidden(out)
        out = self.output_layer(out)
        return out

    def add_training_info(self, train_info):
        self.model_info["train"] = train_info



