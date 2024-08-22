import torch.nn as nn
from .siren import SirenLayer, Sine


class RandomFourierFeatures(nn.Module):
    """Random Fourier Features with default Pytorch initialization.

    Args:
        dim_in (int): Dimension of input.
        dim_out (int): Dimension of output.
        use_bias (bool): Whether to learn bias in linear layer.
        w0 (float):
    """
    def __init__(self,
            dim_in,
            dim_out,
            use_bias=True,
            w0=30.0
                 ):
        super().__init__()  
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.use_bias = use_bias
        self.act_fn = Sine(w0=w0)
        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)

    def forward(self, x):
        return self.act_fn(self.linear(x))

        
class SCONE(nn.Module):
    """Spatially-Collaged Coordinated Network (SCONE) model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden units.
        dim_out (int): Dimension of output.
        use_bias (bool): Whether to learn bias in linear layer.
        w0 (float):
    """
    def __init__(
            self,
            dim_in,
            dim_out,
            scone_configs=None
                 ):
        super().__init__()

        self.dim_in = dim_in
        self.dim_hidden = scone_configs.dim_hidden
        self.dim_out = dim_out
        self.num_layers = scone_configs.num_layers
        self.omegas = scone_configs.omegas
        self.w0 = scone_configs.w0
        self.use_bias = scone_configs.use_bias

        layers = []
        for ind in range(self.num_layers - 1):
            is_first = ind == 0
            layer_dim_in = dim_in if is_first else self.dim_hidden

            layers.append(
                SirenLayer(
                    dim_in=layer_dim_in,
                    dim_out=self.dim_hidden,
                    w0=self.w0,
                    use_bias=self.use_bias,
                is_first=is_first,
                )
            )

        self.net = nn.Sequential(*layers)

        self.last_layer = SirenLayer(
            dim_in=self.dim_hidden, dim_out=dim_out, w0=self.w0, use_bias=self.use_bias, is_last=True
        )

        self.rffs = nn.ModuleList(
            [RandomFourierFeatures(dim_in=dim_in, dim_out=self.dim_hidden, w0=self.omegas[i]) 
             for i in range(self.num_layers-1)]) 

    def forward(self, x, step=None, exp_name=None, **kwargs):
        input = x
        for i, module in enumerate(self.net):
            bases = self.rffs[i](input)
            masks = module(x)
            masks = masks*masks # sin^2
            x = bases*masks

        x = self.last_layer(x)
        return x
        