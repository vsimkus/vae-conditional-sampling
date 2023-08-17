from math import prod
from typing import Callable, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATIONS = {
    'relu': F.relu,
    'leaky_relu': F.leaky_relu,
    'tanh': torch.tanh,
    'sigmoid': F.sigmoid
}


class FullyConnectedNetwork(nn.Module):
    """
    Fully-connected neural network

    Args:
        layer_dims: Dimensions of the model.
        activation: Callable activation function.
    """
    def __init__(self, layer_dims: List[int],
                 *,
                 activation: Union[Callable, str] = F.relu):
        super().__init__()

        self.layer_dims = layer_dims
        if isinstance(activation, Callable):
            self.activation = activation
        elif isinstance(activation, str):
            self.activation = ACTIVATIONS[activation]
        else:
            raise NotImplementedError()

        # Create the linear layers
        linear_layers = []
        for i in range(len(layer_dims)-1):
            linear_layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
        # Create the neural model
        self.linear_layers = nn.ModuleList(linear_layers)

    @property
    def input_dim(self):
        return self.layer_dims[0]

    def forward(self, inputs: torch.Tensor):
        out = inputs
        for l, layer in enumerate(self.linear_layers):
            out = layer(out)
            # No activation after final linear layer
            if l < len(self.linear_layers)-1:
                out = self.activation(out)

        return out


class ResidualBlock(nn.Module):
    """A general-purpose residual block. Works only with 1-dim inputs."""

    def __init__(
        self,
        features: int,
        *,
        activation: Callable = F.relu,
        dropout_probability: float = 0.0,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False
    ):
        super().__init__()
        self.activation = activation
        assert not (use_batch_norm and use_layer_norm), \
            'Cannot use both batch and layer normalization.'

        self.linear_layers = nn.ModuleList(
            [nn.Linear(features, features) for _ in range(2)]
        )
        self.dropout = nn.Dropout(p=dropout_probability)
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(features)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(features)

    def forward(self, inputs: torch.Tensor):
        input_shape = inputs.shape
        temps = self.linear_layers[0](inputs)
        if hasattr(self, 'use_batch_norm') and self.use_batch_norm:
            # Make sure dimension is 2d
            temps = torch.reshape(temps, (-1, temps.shape[-1]))
            temps = self.batch_norm(temps)
            temps = torch.reshape(temps, input_shape[:-1]+(temps.shape[-1],))
        if hasattr(self, 'use_layer_norm') and self.use_layer_norm:
            # Make sure dimension is 2d
            temps = torch.reshape(temps, (-1, temps.shape[-1]))
            temps = self.layer_norm(temps)
            temps = torch.reshape(temps, input_shape[:-1]+(temps.shape[-1],))
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        return inputs + temps


class ResidualFCNetwork(nn.Module):
    """
    Residual fully-connected neural network.

    Args:
        input_dim:              The dimensionality of the inputs.
        output_dim:             The dimensionality of the outputs.
        num_residual_blocks:    The number of full residual blocks in the model.
        residual_block_dim:     The residual block dimensionality.
        activation:             Callable activation function.
        dropout_probability:    Dropout probability in residual blocks.
        use_batch_norm:         Whether to use batch-norm in resblocks or not.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_residual_blocks: int,
                 residual_block_dim: int,
                 *,
                 activation: Union[Callable, str] = F.relu,
                 dropout_probability: float = 0.0,
                 use_batch_norm: bool = False,
                 use_layer_norm: bool = False,
                 ):
        super().__init__()

        self.input_dims = input_dim
        self.output_dim = output_dim
        self.num_residual_blocks = num_residual_blocks
        self.residual_block_dim = residual_block_dim
        if isinstance(activation, Callable):
            self.activation = activation
        elif isinstance(activation, str):
            self.activation = ACTIVATIONS[activation]
        else:
            raise NotImplementedError()
        self.dropout_probability = dropout_probability

        # Add initial layer
        self.initial_layer = nn.Linear(input_dim, residual_block_dim)

        # Create residual blocks
        blocks = [ResidualBlock(residual_block_dim,
                                activation=self.activation,
                                dropout_probability=self.dropout_probability,
                                use_batch_norm=use_batch_norm,
                                use_layer_norm=use_layer_norm)
                  for _ in range(num_residual_blocks)]
        self.blocks = nn.ModuleList(blocks)

        # Add final layer
        self.final_layer = nn.Linear(residual_block_dim, output_dim)

    @property
    def input_dim(self):
        return self.input_dims

    def forward(self, inputs: torch.Tensor):
        out = self.initial_layer(inputs)
        out = self.activation(out)
        for block in self.blocks:
            out = block(out)
            out = self.activation(out)
        return self.final_layer(out)

    def reset_first_n_blocks(self, n):
        @torch.no_grad()
        def weight_reset(m: nn.Module):
            # - check if the current module has reset_parameters & if it's callabed called it on m
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()

        # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        for b in self.blocks[:n]:
            b.apply(fn=weight_reset)


class Conv2DResidualBlock(nn.Module):
    """A general-purpose residual block with 2d convolution."""
    def __init__(
        self,
        channels: int,
        kernel_size: Union[int, Tuple[int]],
        *,
        activation: Callable = F.relu,
        dropout_probability: float = 0.0,
        use_batch_norm: bool = False,
    ):
        super().__init__()
        self.activation = activation

        self.linear_layers = nn.ModuleList(
            [ nn.Conv2d(in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        padding='same')
              for _ in range(2)
            ]
        )
        self.dropout = nn.Dropout2d(p=dropout_probability)
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor):
        input_shape = inputs.shape
        temps = self.linear_layers[0](inputs)
        if self.use_batch_norm:
            # Flatten extra dimensions
            temps = torch.reshape(temps, (-1,) + temps.shape[-3:])
            temps = self.batch_norm(temps)
            temps = torch.reshape(temps, input_shape[:-3]+temps.shape[-3:])
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        return inputs + temps

class ResidualConv2DEncoder(nn.Module):
    """
    Residual convolutional neural network encoder.

    Args:
        input_shape:                 The dimensionality of the inputs.
        input_kernel:                Kernel size of the input layer.
        input_stride:                Stride of the input layer.
        output_dim:                  The dimensionality of the outputs.
        num_residual_blocks:         The number of full residual blocks in the model.
        residual_block_channels:     The residual block dimensionality.
        residual_block_kernel:       Kernel size of the residual blocks (uses same padding).
        activation:                  Callable activation function.
        dropout_probability:         Dropout probability in residual blocks.
        use_batch_norm:              Whether to use batch-norm in resblocks or not.
    """
    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 input_kernel: Union[int, Tuple[int, int]],
                 input_stride: int,
                 output_dim: int,
                 num_residual_blocks: int,
                 residual_block_channels: int,
                 residual_block_kernel: Union[int, Tuple[int, int]],
                 *,
                 activation: Union[Callable, str] = F.relu,
                 dropout_probability: float = 0.0,
                 use_batch_norm: bool = False,
                 ):
        super().__init__()
        assert len(input_shape) == 3

        self.input_shape = input_shape
        self.output_dim = output_dim
        self.input_stride = input_stride
        self.num_residual_blocks = num_residual_blocks
        self.residual_block_channels = residual_block_channels
        if isinstance(activation, Callable):
            self.activation = activation
        elif isinstance(activation, str):
            self.activation = ACTIVATIONS[activation]
        else:
            raise NotImplementedError()
        self.dropout_probability = dropout_probability

        # Add initial layer
        temp = torch.zeros(1, *input_shape)
        print('1', temp.shape)
        self.initial_layer = nn.Conv2d(in_channels=input_shape[0],
                                       out_channels=residual_block_channels,
                                       kernel_size=input_kernel,
                                       stride=input_stride,
                                       padding='valid')
        temp = self.initial_layer(temp)
        print('2', temp.shape)

        # Create residual blocks
        blocks = [Conv2DResidualBlock(channels=residual_block_channels,
                                      kernel_size=residual_block_kernel,
                                      activation=self.activation,
                                      dropout_probability=self.dropout_probability,
                                      use_batch_norm=use_batch_norm)
                  for _ in range(num_residual_blocks)]
        self.blocks = nn.ModuleList(blocks)
        # temp = self.blocks(temp)
        temp = self.blocks[0](temp)
        print('3', temp.shape)

        # Add final layer
        self.final_layer = nn.Linear(prod(temp.shape), output_dim)

    def forward(self, inputs: torch.Tensor):
        inputs = inputs.view(inputs.shape[:-1] + self.input_shape)

        out = self.initial_layer(inputs)
        out = self.activation(out)
        for block in self.blocks:
            out = block(out)
            out = self.activation(out)

        out = out.view(*inputs.shape[:-3], -1)
        return self.final_layer(out)

class ResidualConv2DDecoder(nn.Module):
    """
    Residual convolutional neural network decoder.

    Args:
        input_dim:                   The dimensionality of the inputs.
        input_upsample_scale:
        input_kernel:                Kernel size of the input layer.
        output_shape:                  The dimensionality of the outputs.
        output_layer_outpadding:
        output_layer_stride:
        num_residual_blocks:         The number of full residual blocks in the model.
        residual_block_channels:     The residual block dimensionality.
        residual_block_kernel:       Kernel size of the residual blocks (uses same padding).
        activation:                  Callable activation function.
        dropout_probability:         Dropout probability in residual blocks.
        use_batch_norm:              Whether to use batch-norm in resblocks or not.
    """
    def __init__(self,
                 input_dim: int,
                 input_upsample_scale: int,
                 output_shape: Tuple[int, int, int],
                 output_layer_outpadding: int,
                 output_layer_stride: int,
                 output_layer_kernel: Tuple[int, int],
                 num_residual_blocks: int,
                 residual_block_channels: int,
                 residual_block_kernel: Union[int, Tuple[int, int]],
                 *,
                 activation: Union[Callable, str] = F.relu,
                 dropout_probability: float = 0.0,
                 use_batch_norm: bool = False,
                 ):
        super().__init__()

        self.input_dim = input_dim
        self.output_shape = output_shape
        self.num_residual_blocks = num_residual_blocks
        self.residual_block_channels = residual_block_channels
        self.input_upsample_scale = input_upsample_scale
        self.output_layer_outpadding = output_layer_outpadding
        if isinstance(activation, Callable):
            self.activation = activation
        elif isinstance(activation, str):
            self.activation = ACTIVATIONS[activation]
        else:
            raise NotImplementedError()
        self.dropout_probability = dropout_probability

        # Add initial layer
        self.initial_layer = nn.Linear(in_features=input_dim,
                                       out_features=residual_block_channels)
        self.up = nn.Upsample(scale_factor=input_upsample_scale)

        # Create residual blocks
        blocks = [Conv2DResidualBlock(channels=residual_block_channels,
                                      kernel_size=residual_block_kernel,
                                      activation=self.activation,
                                      dropout_probability=self.dropout_probability,
                                      use_batch_norm=use_batch_norm)
                  for _ in range(num_residual_blocks)]
        self.blocks = nn.ModuleList(blocks)

        # Add final layer
        self.final_layer = nn.ConvTranspose2d(in_channels=self.residual_block_channels,
                                              out_channels=self.output_shape[0],
                                              stride=output_layer_stride,
                                              kernel_size=output_layer_kernel,
                                              output_padding=output_layer_outpadding)

    def forward(self, inputs: torch.Tensor):
        out = self.initial_layer(inputs)
        out = out.view(*out.shape, 1, 1)
        out = out.view(-1, *(out.shape[-3:]))
        out = self.up(out)

        out = self.activation(out)
        for block in self.blocks:
            out = block(out)
            out = self.activation(out)

        out = self.final_layer(out)
        out = out.view(*inputs.shape[:-1], -1)
        return out
