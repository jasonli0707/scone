import torch.nn as nn
import rff


class MLP(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        mlp_configs=None
    ):
        
        super().__init__()
        self.mlp_configs = mlp_configs
        self.encoding_configs = mlp_configs.encoding
        num_layers = mlp_configs.num_layers
        dim_hidden = mlp_configs.dim_hidden
        use_bias = mlp_configs.use_bias

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.use_bias = use_bias

        if self.encoding_configs.type == 'basic':
            self.encoding = rff.layers.BasicEncoding()
            enc_size = dim_in
        elif self.encoding_configs.type == 'positional':
            self.encoding = rff.layers.PositionalEncoding(sigma=self.encoding_configs.sigma, m=self.encoding_configs.encoded_size)
            enc_size = int(2*self.encoding_configs.encoded_size*dim_in)
        elif self.encoding_configs.type == 'gaussian':
            self.encoding = rff.layers.GaussianEncoding(sigma=self.encoding_configs.sigma, input_size=dim_in, encoded_size=self.encoding_configs.encoded_size)
            enc_size = int(2*self.encoding_configs.encoded_size)

        self.enc_size = enc_size

        layers = []
        for i in range(num_layers - 1):
            if i==0:
                layers.append(nn.Linear(enc_size, dim_hidden, bias=use_bias))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(dim_hidden, dim_hidden, bias=use_bias))
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)
        self.last_layer = nn.Linear(dim_hidden, dim_out, bias=use_bias)

    def forward(self, x, step=None, exp_name=None):
        x = self.encoding(x)
        x = self.net(x)
        return self.last_layer(x)