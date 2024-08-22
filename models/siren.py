import torch
from torch import nn
from math import sqrt


class Sine(nn.Module):
    """Sine activation with scaling.

    Args:
        w0 (float): Omega_0 parameter from SIREN paper.
    """

    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Module):
    """Implements a single SIREN layer.

    Args:
        dim_in (int): Dimension of input.
        dim_out (int): Dimension of output.
        w0 (float):
        c (float): c value from SIREN paper used for weight initialization.
        is_first (bool): Whether this is first layer of model.
        is_last (bool): Whether this is last layer of model. If it is, no
            activation is applied and 0.5 is added to the output. Since we
            assume all training data lies in [0, 1], this allows for centering
            the output of the model.
        use_bias (bool): Whether to learn bias in linear layer.
        activation (torch.nn.Module): Activation function. If None, defaults to
            Sine activation.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        w0=30.0,
        c=6.0,
        is_first=False,
        is_last=False,
        use_bias=True,
        activation=None,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.is_first = is_first
        self.is_last = is_last

        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)

        # Initialize layers following SIREN paper
        w_std = (1 / dim_in) if self.is_first else (sqrt(c / dim_in) / w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        if use_bias:
            nn.init.uniform_(self.linear.bias, -w_std, w_std)

        self.activation = Sine(w0) if activation is None else activation

    def forward(self, x):
        out = self.linear(x)
        if not self.is_last:
            out = self.activation(out)
        return out


class Siren(nn.Module):
    """SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        out_range,
        siren_configs
    ):
        super().__init__()
        self.siren_configs = siren_configs
        num_layers = siren_configs.num_layers
        dim_hidden = siren_configs.dim_hidden
        w0 = siren_configs.w0
        w0_initial = siren_configs.w0_initial
        use_bias = siren_configs.use_bias
        
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        layers = []
        for ind in range(num_layers - 1):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            layers.append(
                SirenLayer(
                    dim_in=layer_dim_in,
                    dim_out=dim_hidden,
                    w0=layer_w0,
                    use_bias=use_bias,
                    is_first=is_first,
                )
            )

        self.net = nn.Sequential(*layers)

        self.last_layer = SirenLayer(
            dim_in=dim_hidden, dim_out=dim_out, w0=w0, use_bias=use_bias, is_last=True
        )

    def forward(self, raw_coords, *args, **kwargs):
        """Forward pass of SIREN model.

        Args:
            raw_coords (torch.Tensor): Tensor of shape (*, dim_in), where * means any
                number of dimensions.
                ** Note that the coordinates are not yet normalized to [-1, 1] **

        Returns:
            Tensor of shape (*, dim_out).
        """
        x = self.net(raw_coords)
        return self.last_layer(x)