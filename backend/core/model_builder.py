"""
Neural Network Builder.

Builds nn.Module models from a JSON layer specification.
This powers the visual model builder in the frontend.

Each layer is a dict:
    {"type": "linear", "in_features": 784, "out_features": 256}
    {"type": "conv2d", "in_channels": 1, "out_channels": 32, "kernel_size": 3, "padding": 1}
    {"type": "relu"}
    {"type": "flatten"}
    ...

Supports:
- All standard PyTorch layers
- Automatic shape inference
- Export to code / JSON
- Validation of layer compatibility
"""

import json
import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ── Layer registry ──

LAYER_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Linear
    'linear': {
        'class': nn.Linear,
        'params': {'in_features': 'int', 'out_features': 'int', 'bias': 'bool'},
        'defaults': {'bias': True},
        'category': 'linear',
    },
    # Convolution
    'conv2d': {
        'class': nn.Conv2d,
        'params': {'in_channels': 'int', 'out_channels': 'int', 'kernel_size': 'int',
                   'stride': 'int', 'padding': 'int', 'bias': 'bool'},
        'defaults': {'stride': 1, 'padding': 0, 'bias': True},
        'category': 'convolution',
    },
    'conv1d': {
        'class': nn.Conv1d,
        'params': {'in_channels': 'int', 'out_channels': 'int', 'kernel_size': 'int',
                   'stride': 'int', 'padding': 'int'},
        'defaults': {'stride': 1, 'padding': 0},
        'category': 'convolution',
    },
    'conv_transpose2d': {
        'class': nn.ConvTranspose2d,
        'params': {'in_channels': 'int', 'out_channels': 'int', 'kernel_size': 'int',
                   'stride': 'int', 'padding': 'int'},
        'defaults': {'stride': 1, 'padding': 0},
        'category': 'convolution',
    },
    # Pooling
    'maxpool2d': {
        'class': nn.MaxPool2d,
        'params': {'kernel_size': 'int', 'stride': 'int', 'padding': 'int'},
        'defaults': {'stride': None, 'padding': 0},
        'category': 'pooling',
    },
    'avgpool2d': {
        'class': nn.AvgPool2d,
        'params': {'kernel_size': 'int', 'stride': 'int', 'padding': 'int'},
        'defaults': {'stride': None, 'padding': 0},
        'category': 'pooling',
    },
    'adaptive_avgpool2d': {
        'class': nn.AdaptiveAvgPool2d,
        'params': {'output_size': 'int_or_tuple'},
        'defaults': {},
        'category': 'pooling',
    },
    # Normalization
    'batchnorm1d': {
        'class': nn.BatchNorm1d,
        'params': {'num_features': 'int'},
        'defaults': {},
        'category': 'normalization',
    },
    'batchnorm2d': {
        'class': nn.BatchNorm2d,
        'params': {'num_features': 'int'},
        'defaults': {},
        'category': 'normalization',
    },
    'layernorm': {
        'class': nn.LayerNorm,
        'params': {'normalized_shape': 'int_or_list'},
        'defaults': {},
        'category': 'normalization',
    },
    'groupnorm': {
        'class': nn.GroupNorm,
        'params': {'num_groups': 'int', 'num_channels': 'int'},
        'defaults': {},
        'category': 'normalization',
    },
    # Activations
    'relu': {'class': nn.ReLU, 'params': {}, 'defaults': {}, 'category': 'activation'},
    'leaky_relu': {
        'class': nn.LeakyReLU,
        'params': {'negative_slope': 'float'},
        'defaults': {'negative_slope': 0.01},
        'category': 'activation',
    },
    'gelu': {'class': nn.GELU, 'params': {}, 'defaults': {}, 'category': 'activation'},
    'silu': {'class': nn.SiLU, 'params': {}, 'defaults': {}, 'category': 'activation'},
    'sigmoid': {'class': nn.Sigmoid, 'params': {}, 'defaults': {}, 'category': 'activation'},
    'tanh': {'class': nn.Tanh, 'params': {}, 'defaults': {}, 'category': 'activation'},
    'softmax': {
        'class': nn.Softmax,
        'params': {'dim': 'int'},
        'defaults': {'dim': -1},
        'category': 'activation',
    },
    # Regularization
    'dropout': {
        'class': nn.Dropout,
        'params': {'p': 'float'},
        'defaults': {'p': 0.5},
        'category': 'regularization',
    },
    'dropout2d': {
        'class': nn.Dropout2d,
        'params': {'p': 'float'},
        'defaults': {'p': 0.5},
        'category': 'regularization',
    },
    # Reshape
    'flatten': {
        'class': nn.Flatten,
        'params': {'start_dim': 'int', 'end_dim': 'int'},
        'defaults': {'start_dim': 1, 'end_dim': -1},
        'category': 'reshape',
    },
    # Recurrent
    'lstm': {
        'class': nn.LSTM,
        'params': {'input_size': 'int', 'hidden_size': 'int', 'num_layers': 'int',
                   'batch_first': 'bool', 'dropout': 'float', 'bidirectional': 'bool'},
        'defaults': {'num_layers': 1, 'batch_first': True, 'dropout': 0.0, 'bidirectional': False},
        'category': 'recurrent',
    },
    'gru': {
        'class': nn.GRU,
        'params': {'input_size': 'int', 'hidden_size': 'int', 'num_layers': 'int',
                   'batch_first': 'bool', 'dropout': 'float', 'bidirectional': 'bool'},
        'defaults': {'num_layers': 1, 'batch_first': True, 'dropout': 0.0, 'bidirectional': False},
        'category': 'recurrent',
    },
    # Transformer
    'multihead_attention': {
        'class': nn.MultiheadAttention,
        'params': {'embed_dim': 'int', 'num_heads': 'int', 'dropout': 'float', 'batch_first': 'bool'},
        'defaults': {'dropout': 0.0, 'batch_first': True},
        'category': 'transformer',
    },
    'transformer_encoder_layer': {
        'class': nn.TransformerEncoderLayer,
        'params': {'d_model': 'int', 'nhead': 'int', 'dim_feedforward': 'int',
                   'dropout': 'float', 'batch_first': 'bool'},
        'defaults': {'dim_feedforward': 2048, 'dropout': 0.1, 'batch_first': True},
        'category': 'transformer',
    },
    # Embedding
    'embedding': {
        'class': nn.Embedding,
        'params': {'num_embeddings': 'int', 'embedding_dim': 'int'},
        'defaults': {},
        'category': 'embedding',
    },
}


# ── Builder ──

def build_layer(spec: Dict[str, Any]) -> nn.Module:
    """Build a single layer from a spec dict."""
    layer_type = spec.get('type', '').lower()
    if layer_type not in LAYER_REGISTRY:
        raise ValueError(f"Unknown layer type: '{layer_type}'. Available: {list(LAYER_REGISTRY.keys())}")

    reg = LAYER_REGISTRY[layer_type]
    cls = reg['class']

    # Collect constructor args
    kwargs = {}
    for param_name, param_type in reg['params'].items():
        if param_name in spec:
            kwargs[param_name] = spec[param_name]
        elif param_name in reg['defaults']:
            val = reg['defaults'][param_name]
            if val is not None:
                kwargs[param_name] = val

    return cls(**kwargs)


class BuiltModel(nn.Module):
    """A model built from a list of layer specs."""

    def __init__(self, layer_specs: List[Dict[str, Any]], name: str = 'CustomModel'):
        super().__init__()
        self.layer_specs = layer_specs
        self.model_name = name

        layers = []
        for i, spec in enumerate(layer_specs):
            layer = build_layer(spec)
            layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, (nn.LSTM, nn.GRU)):
                x, _ = layer(x)
            elif isinstance(layer, nn.MultiheadAttention):
                x, _ = layer(x, x, x)
            else:
                x = layer(x)
        return x


def build_model(spec: Dict[str, Any]) -> nn.Module:
    """
    Build a model from a full spec:
    {
        "name": "MyModel",
        "layers": [
            {"type": "conv2d", "in_channels": 1, "out_channels": 32, "kernel_size": 3, "padding": 1},
            {"type": "relu"},
            ...
        ]
    }
    """
    layers = spec.get('layers', [])
    name = spec.get('name', 'CustomModel')
    return BuiltModel(layers, name=name)


def validate_model_spec(spec: Dict[str, Any]) -> List[str]:
    """Validate a model spec, return list of errors."""
    errors = []
    layers = spec.get('layers', [])

    if not layers:
        errors.append("Model has no layers")
        return errors

    for i, layer_spec in enumerate(layers):
        lt = layer_spec.get('type', '')
        if lt not in LAYER_REGISTRY:
            errors.append(f"Layer {i}: unknown type '{lt}'")
            continue

        reg = LAYER_REGISTRY[lt]
        for param_name, param_type in reg['params'].items():
            if param_name not in layer_spec and param_name not in reg['defaults']:
                errors.append(f"Layer {i} ({lt}): missing required param '{param_name}'")

    return errors


def infer_shapes(spec: Dict[str, Any], input_shape: Tuple[int, ...]) -> List[Dict[str, Any]]:
    """
    Try to infer output shapes for each layer given an input shape.
    Returns annotated layer list with 'output_shape' added.
    """
    model = build_model(spec)
    annotated = copy.deepcopy(spec.get('layers', []))

    try:
        x = torch.randn(1, *input_shape)
        for i, layer in enumerate(model.layers):
            if isinstance(layer, (nn.LSTM, nn.GRU)):
                x, _ = layer(x)
            elif isinstance(layer, nn.MultiheadAttention):
                x, _ = layer(x, x, x)
            else:
                x = layer(x)
            annotated[i]['output_shape'] = list(x.shape[1:])  # drop batch dim
    except Exception as e:
        # Mark where it failed
        for j in range(len(annotated)):
            if 'output_shape' not in annotated[j]:
                annotated[j]['output_shape'] = None
                if j == len(annotated) - 1 or (j > 0 and annotated[j-1].get('output_shape') is not None):
                    annotated[j]['shape_error'] = str(e)

    return annotated


def model_spec_to_code(spec: Dict[str, Any]) -> str:
    """Generate Python code from a model spec."""
    name = spec.get('name', 'CustomModel')
    layers = spec.get('layers', [])

    lines = [
        'import torch',
        'import torch.nn as nn',
        '',
        f'class {name}(nn.Module):',
        '    def __init__(self):',
        '        super().__init__()',
    ]

    for i, layer_spec in enumerate(layers):
        lt = layer_spec.get('type', '')
        if lt not in LAYER_REGISTRY:
            continue
        reg = LAYER_REGISTRY[lt]
        cls_name = reg['class'].__name__

        params = []
        for pname in reg['params']:
            if pname in layer_spec:
                val = layer_spec[pname]
                params.append(f'{pname}={repr(val)}')
            elif pname in reg['defaults'] and reg['defaults'][pname] is not None:
                params.append(f'{pname}={repr(reg["defaults"][pname])}')

        lines.append(f'        self.layer_{i} = nn.{cls_name}({", ".join(params)})')

    lines.append('')
    lines.append('    def forward(self, x):')
    for i, layer_spec in enumerate(layers):
        lt = layer_spec.get('type', '')
        if lt in ('lstm', 'gru'):
            lines.append(f'        x, _ = self.layer_{i}(x)')
        elif lt == 'multihead_attention':
            lines.append(f'        x, _ = self.layer_{i}(x, x, x)')
        else:
            lines.append(f'        x = self.layer_{i}(x)')
    lines.append('        return x')

    return '\n'.join(lines)


def get_layer_catalog() -> Dict[str, Any]:
    """Return the full layer catalog for the frontend UI."""
    catalog = {}
    for name, reg in LAYER_REGISTRY.items():
        catalog[name] = {
            'category': reg['category'],
            'params': reg['params'],
            'defaults': reg['defaults'],
        }
    return catalog


# ── Presets ──

MODEL_PRESETS = {
    'mnist_cnn': {
        'name': 'MNIST_CNN',
        'description': 'Simple CNN for MNIST (28x28 grayscale)',
        'input_shape': [1, 28, 28],
        'layers': [
            {'type': 'conv2d', 'in_channels': 1, 'out_channels': 32, 'kernel_size': 3, 'padding': 1},
            {'type': 'relu'},
            {'type': 'maxpool2d', 'kernel_size': 2},
            {'type': 'conv2d', 'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'padding': 1},
            {'type': 'relu'},
            {'type': 'maxpool2d', 'kernel_size': 2},
            {'type': 'flatten'},
            {'type': 'linear', 'in_features': 3136, 'out_features': 128},
            {'type': 'relu'},
            {'type': 'dropout', 'p': 0.5},
            {'type': 'linear', 'in_features': 128, 'out_features': 10},
        ],
    },
    'mnist_mlp': {
        'name': 'MNIST_MLP',
        'description': 'Simple MLP for MNIST',
        'input_shape': [1, 28, 28],
        'layers': [
            {'type': 'flatten'},
            {'type': 'linear', 'in_features': 784, 'out_features': 512},
            {'type': 'relu'},
            {'type': 'dropout', 'p': 0.2},
            {'type': 'linear', 'in_features': 512, 'out_features': 256},
            {'type': 'relu'},
            {'type': 'dropout', 'p': 0.2},
            {'type': 'linear', 'in_features': 256, 'out_features': 10},
        ],
    },
    'cifar10_cnn': {
        'name': 'CIFAR10_CNN',
        'description': 'CNN for CIFAR-10 (32x32 RGB)',
        'input_shape': [3, 32, 32],
        'layers': [
            {'type': 'conv2d', 'in_channels': 3, 'out_channels': 32, 'kernel_size': 3, 'padding': 1},
            {'type': 'batchnorm2d', 'num_features': 32},
            {'type': 'relu'},
            {'type': 'conv2d', 'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'padding': 1},
            {'type': 'batchnorm2d', 'num_features': 64},
            {'type': 'relu'},
            {'type': 'maxpool2d', 'kernel_size': 2},
            {'type': 'conv2d', 'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'padding': 1},
            {'type': 'batchnorm2d', 'num_features': 128},
            {'type': 'relu'},
            {'type': 'maxpool2d', 'kernel_size': 2},
            {'type': 'flatten'},
            {'type': 'linear', 'in_features': 8192, 'out_features': 256},
            {'type': 'relu'},
            {'type': 'dropout', 'p': 0.5},
            {'type': 'linear', 'in_features': 256, 'out_features': 10},
        ],
    },
    'text_classifier': {
        'name': 'TextClassifier',
        'description': 'Simple text classification model',
        'input_shape': [100],
        'layers': [
            {'type': 'embedding', 'num_embeddings': 10000, 'embedding_dim': 128},
            {'type': 'lstm', 'input_size': 128, 'hidden_size': 256, 'num_layers': 2, 'batch_first': True, 'dropout': 0.3},
            {'type': 'flatten'},
            {'type': 'linear', 'in_features': 25600, 'out_features': 128},
            {'type': 'relu'},
            {'type': 'linear', 'in_features': 128, 'out_features': 5},
        ],
    },
    'autoencoder': {
        'name': 'Autoencoder',
        'description': 'Simple autoencoder for 28x28 images',
        'input_shape': [1, 28, 28],
        'layers': [
            {'type': 'flatten'},
            {'type': 'linear', 'in_features': 784, 'out_features': 256},
            {'type': 'relu'},
            {'type': 'linear', 'in_features': 256, 'out_features': 64},
            {'type': 'relu'},
            {'type': 'linear', 'in_features': 64, 'out_features': 256},
            {'type': 'relu'},
            {'type': 'linear', 'in_features': 256, 'out_features': 784},
            {'type': 'sigmoid'},
        ],
    },
}
