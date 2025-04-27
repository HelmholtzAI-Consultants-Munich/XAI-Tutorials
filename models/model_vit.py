############################################
# imports
############################################

import math
import torch
import torch.nn as nn

from functools import partial

import warnings

warnings.filterwarnings("ignore")


############################################
# Model classes
############################################


class PatchEmbed(nn.Module):
    """
    Converts an input image into a sequence of patch embeddings using a convolutional projection.

    This module is commonly used in Vision Transformers (ViTs) to split an image into non-overlapping patches,
    flatten them, and project each patch into a high-dimensional embedding space.

    :param img_size: Size of the input image (assumes square image). Default is 224.
    :type img_size: int
    :param patch_size: Size of each image patch (assumes square patch). Default is 16.
    :type patch_size: int
    :param in_chans: Number of input channels (e.g., 3 for RGB). Default is 3.
    :type in_chans: int
    :param embed_dim: Dimension of the embedding for each patch. Default is 768.
    :type embed_dim: int
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        """Constructor for PatchEmbed class"""
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        Forward pass that projects an image into a sequence of flattened patch embeddings.

        :param x: Input tensor of shape (B, C, H, W), where B is batch size, C is channels, H and W are spatial dimensions.
        :type x: torch.Tensor
        :return: Patch embeddings of shape (B, num_patches, embed_dim).
        :rtype: torch.Tensor
        """
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class DropPath(nn.Module):
    """
    DropPath module applies stochastic depth regularization to input tensors.

    This module randomly drops entire paths (such as residual connections) during training,
    which helps to regularize deep residual networks and improve generalization.

    :param drop_prob: Probability of dropping the path. Must be between 0 and 1.
    :type drop_prob: float
    """

    def __init__(self, drop_prob=None):
        """Constructor for DropPath class"""
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """
        Forward pass of the DropPath module.

        Applies stochastic depth to the input tensor `x` based on the training mode.

        :param x: Input tensor to apply drop path to.
        :type x: torch.Tensor
        :return: Output tensor after applying stochastic depth.
        :rtype: torch.Tensor
        """
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """
    Multi-Layer Perceptron (MLP) module with configurable input, hidden, and output dimensions.

    This module consists of two linear layers with an activation function and dropout layers.

    :param in_features: Number of input features.
    :type in_features: int
    :param hidden_features: Number of features in the hidden layer. Defaults to `in_features` if None.
    :type hidden_features: int or None
    :param out_features: Number of output features. Defaults to `in_features` if None.
    :type out_features: int or None
    :param act_layer: Activation layer class to apply after the first linear transformation.
    :type act_layer: nn.Module
    :param drop: Dropout probability applied after each linear layer.
    :type drop: float
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        """Constructor for Mlp class"""
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Forward pass of the MLP.

        Applies linear transformation, activation, dropout, and a second linear transformation
        followed by another dropout.

        :param x: Input tensor of shape (batch_size, in_features).
        :type x: torch.Tensor
        :return: Output tensor of shape (batch_size, out_features).
        :rtype: torch.Tensor
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """
    Multi-head self-attention mechanism as used in transformer architectures.

    :param dim: Dimension of input and output embeddings.
    :type dim: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param qkv_bias: If True, adds a learnable bias to query, key, and value.
    :type qkv_bias: bool
    :param qk_scale: Override default scaling of attention scores (1/sqrt(head_dim)).
    :type qk_scale: float or None
    :param attn_drop: Dropout rate for attention probabilities.
    :type attn_drop: float
    :param proj_drop: Dropout rate after final projection.
    :type proj_drop: float
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        """Constructor for Attention class"""
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        Compute attention and apply to values.

        :param x: Input tensor of shape (B, N, C), where B is batch size, N is number of tokens, and C is embedding dimension.
        :type x: torch.Tensor
        :return: Tuple of output tensor and attention weights.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class AttentionBlock(nn.Module):
    """
    Transformer-style attention block combining multi-head self-attention, MLP, layer normalization, and dropout.

    :param dim: Input and output embedding dimension.
    :type dim: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param mlp_ratio: Expansion ratio for MLP hidden dimension.
    :type mlp_ratio: float
    :param qkv_bias: Whether to include bias terms in query, key, value projections.
    :type qkv_bias: bool
    :param qk_scale: Optional scaling factor for attention scores.
    :type qk_scale: float or None
    :param drop: Dropout rate after projections and MLP.
    :type drop: float
    :param attn_drop: Dropout rate applied to attention scores.
    :type attn_drop: float
    :param drop_path: Drop path rate for stochastic depth.
    :type drop_path: float
    :param act_layer: Activation function used in MLP.
    :type act_layer: nn.Module
    :param norm_layer: Normalization layer applied before attention and MLP.
    :type norm_layer: nn.Module
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        """Constructor for AttentionBlock class"""
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        """
        Forward pass of the attention block.

        :param x: Input tensor of shape (B, N, C).
        :type x: torch.Tensor
        :param return_attention: If True, return attention weights instead of block output.
        :type return_attention: bool
        :return: Block output tensor or attention weights.
        :rtype: Union[torch.Tensor, torch.Tensor]
        """
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) model for image classification tasks.

    This class implements a transformer architecture that splits the input image into patches,
    embeds them, adds positional encoding, and applies multiple attention blocks before classification.

    :param img_size: Size of input images (height/width).
    :type img_size: list[int]
    :param patch_size: Size of each image patch.
    :type patch_size: int
    :param in_chans: Number of input image channels (e.g., 3 for RGB).
    :type in_chans: int
    :param num_classes: Number of output classes for classification. If 0, no classification head is used.
    :type num_classes: int
    :param embed_dim: Embedding dimension for patch embeddings.
    :type embed_dim: int
    :param depth: Number of transformer blocks.
    :type depth: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param mlp_ratio: Ratio between MLP hidden dimension and embedding dimension.
    :type mlp_ratio: float
    :param qkv_bias: If True, adds a bias term to query, key, and value projections.
    :type qkv_bias: bool
    :param qk_scale: Optional scaling factor for query-key dot product.
    :type qk_scale: float or None
    :param drop_rate: Dropout rate applied after patch embedding and attention blocks.
    :type drop_rate: float
    :param attn_drop_rate: Dropout rate for attention weights.
    :type attn_drop_rate: float
    :param drop_path_rate: Drop path rate for stochastic depth.
    :type drop_path_rate: float
    :param norm_layer: Normalization layer used (default is LayerNorm).
    :type norm_layer: torch.nn.Module
    :param kwargs: Additional arguments.
    :type kwargs: dict
    """

    def __init__(
        self,
        img_size=[224],
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        **kwargs,
    ):
        """Constructor for VisionTransformer class"""
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                AttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize model weights.

        :param m: Model module to initialize.
        :type m: torch.nn.Module
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        """
        Interpolate positional encoding to match input image size.

        :param x: Input tensor after patch embedding.
        :type x: torch.Tensor
        :param w: Width of the original image.
        :type w: int
        :param h: Height of the original image.
        :type h: int
        :return: Positional encoding tensor adapted to input dimensions.
        :rtype: torch.Tensor
        """
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        """
        Prepare tokens by patch embedding and adding positional encoding.

        :param x: Input image tensor (batch_size, channels, width, height).
        :type x: torch.Tensor
        :return: Tokens with positional encoding.
        :rtype: torch.Tensor
        """
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x):
        """
        Forward pass through the Vision Transformer.

        :param x: Input image tensor.
        :type x: torch.Tensor
        :return: Output embedding for [CLS] token.
        :rtype: torch.Tensor
        """
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def get_last_selfattention(self, x):
        """
        Retrieve the self-attention map from the last transformer block.

        :param x: Input image tensor.
        :type x: torch.Tensor
        :return: Attention map from the last block.
        :rtype: torch.Tensor
        """
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        """
        Get intermediate layer outputs from the Vision Transformer.

        :param x: Input image tensor.
        :type x: torch.Tensor
        :param n: Number of last layers to return.
        :type n: int
        :return: List of intermediate layer outputs.
        :rtype: list[torch.Tensor]
        """
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


class VitGenerator(object):
    """
    Wrapper class for Vision Transformer (ViT) models, enabling model initialization,
    weight loading, and feature extraction (e.g., attention maps).

    :param name_model: Name of the Vision Transformer model to initialize ("vit_tiny", "vit_small", or "vit_base").
    :type name_model: str
    :param patch_size: Patch size for the Vision Transformer model.
    :type patch_size: int
    :param device: Torch device (e.g., "cpu" or "cuda") on which the model should run.
    :type device: torch.device
    :param evaluate: Whether to freeze the model parameters for evaluation mode (default: True).
    :type evaluate: bool
    :param random: Whether to initialize with random weights instead of pretrained ones (default: False).
    :type random: bool
    :param verbose: Whether to print detailed information during model setup (default: False).
    :type verbose: bool
    """

    def __init__(self, name_model, patch_size, device, evaluate=True, random=False, verbose=False):
        """Constructor for VitGenerator class"""
        self.name_model = name_model
        self.patch_size = patch_size
        self.evaluate = evaluate
        self.device = device
        self.verbose = verbose
        self.model = self._getModel()
        self._initializeModel()
        if not random:
            self._loadPretrainedWeights()

    def _getModel(self):
        """
        Initialize a Vision Transformer model based on the given model name and patch size.

        :return: Initialized Vision Transformer model.
        :rtype: VisionTransformer
        """
        if self.verbose:
            print(f"[INFO] Initializing {self.name_model} with patch size of {self.patch_size}")
        if self.name_model == "vit_tiny":
            model = VisionTransformer(
                patch_size=self.patch_size,
                embed_dim=192,
                depth=12,
                num_heads=3,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )

        elif self.name_model == "vit_small":
            model = VisionTransformer(
                patch_size=self.patch_size,
                embed_dim=384,
                depth=12,
                num_heads=6,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )

        elif self.name_model == "vit_base":
            model = VisionTransformer(
                patch_size=self.patch_size,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
        else:
            raise f"No model found with {self.name_model}"

        return model

    def _initializeModel(self):
        """
        Set model to evaluation mode and move it to the specified device if evaluation is enabled.
        """
        if self.evaluate:
            for p in self.model.parameters():
                p.requires_grad = False

            self.model.eval()

        self.model.to(self.device)

    def _loadPretrainedWeights(self):
        """
        Load pretrained weights into the Vision Transformer model if available.
        If no matching pretrained model is found, random weights are used.
        """
        if self.verbose:
            print("[INFO] Loading weights")
        url = None
        if self.name_model == "vit_small" and self.patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"

        elif self.name_model == "vit_small" and self.patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"

        elif self.name_model == "vit_base" and self.patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"

        elif self.name_model == "vit_base" and self.patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"

        if url is None:
            print(
                f"Since no pretrained weights have been found with name {self.name_model} and patch size {self.patch_size}, random weights will be used"
            )

        else:
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            self.model.load_state_dict(state_dict, strict=True)

    def get_last_selfattention(self, img):
        """
        Retrieve the last self-attention maps from the Vision Transformer for a given input image.

        :param img: Input image tensor.
        :type img: torch.Tensor
        :return: Last self-attention maps of the model.
        :rtype: torch.Tensor
        """
        return self.model.get_last_selfattention(img.to(self.device))

    def __call__(self, x):
        """
        Forward pass through the Vision Transformer model.

        :param x: Input image tensor.
        :type x: torch.Tensor
        :return: Output of the Vision Transformer model.
        :rtype: torch.Tensor
        """
        return self.model(x)


############################################
# helper functions
############################################


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """
    Fills the input tensor with values drawn from a truncated normal distribution.

    Values are effectively drawn from a normal distribution with the specified mean and standard deviation,
    but values outside the range [a, b] are redrawn until they fall within the bounds.

    :param tensor: The tensor to be filled.
    :type tensor: torch.Tensor
    :param mean: The mean of the normal distribution. Default is 0.0.
    :type mean: float
    :param std: The standard deviation of the normal distribution. Default is 1.0.
    :type std: float
    :param a: Minimum cutoff value for truncation. Default is -2.0.
    :type a: float
    :param b: Maximum cutoff value for truncation. Default is 2.0.
    :type b: float
    :return: The input tensor filled with truncated normal values.
    :rtype: torch.Tensor
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """
    Internal method to fill a tensor with truncated normal values without gradient tracking.

    This method uses inverse transform sampling to draw samples from a truncated normal distribution.

    Cut & paste from PyTorch official master until it's in a few official releases - RW
    Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf

    :param tensor: The tensor to be filled.
    :type tensor: torch.Tensor
    :param mean: The mean of the normal distribution.
    :type mean: float
    :param std: The standard deviation of the normal distribution.
    :type std: float
    :param a: Minimum cutoff value for truncation.
    :type a: float
    :param b: Maximum cutoff value for truncation.
    :type b: float
    :return: The input tensor filled with values.
    :rtype: torch.Tensor
    """

    def norm_cdf(x):
        """
        Computes the cumulative distribution function (CDF) for the standard normal distribution.

        :param x: The input value.
        :type x: float
        :return: The CDF evaluated at x.
        :rtype: float
        """
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample during training. Useful for regularization in deep networks.

    Randomly drops entire residual paths during training with a given probability. The output is rescaled
    to maintain the same expected value.

    :param x: Input tensor.
    :type x: torch.Tensor
    :param drop_prob: Probability of dropping paths. Default is 0.0 (no drop).
    :type drop_prob: float
    :param training: Flag indicating whether the model is in training mode.
    :type training: bool
    :return: Tensor with some paths randomly dropped during training.
    :rtype: torch.Tensor
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
