import torch
import torch.nn as nn

from einops import repeat
from einops.layers.torch import Rearrange

class Patch(nn.Module):
    def __init__(self, patch_size):
        super(Patch, self).__init__()
        self.net = Rearrange("b c (h ph) (w pw) -> b (h w) (ph pw c)", ph=patch_size, pw=patch_size)
    
    def forward(self, x):
        x = self.net(x)
        return x

class LinearProjection(nn.Module):
    def __init__(self, patch_dim, dim):
        super(LinearProjection, self).__init__()
        self.net = nn.Linear(patch_dim, dim)

    def forward(self, x):
        x = self.net(x)
        return x

class Embedding(nn.Module):
    def __init__(self, dim, n_patches):
        super(Embedding, self).__init__()
        
        # class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches + 1, dim))

    def forward(self, x):
        batch_size, _, __ = x.shape

        # add cls token
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=batch_size)
        x = torch.concat([cls_tokens, x], dim=1)

        # add positional encoding
        x += self.pos_embedding

        return x

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = self.net(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)

        self.split_into_heads = Rearrange("b n (h d) -> b h n d", h=self.n_heads)

        self.softmax = nn.Softmax(dim=-1)

        self.concat = Rearrange("b h n d -> b n (h d)", h=self.n_heads)

    def forward(self, x):
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        q_heads = self.split_into_heads(q)
        k_heads = self.split_into_heads(k)
        v_heads = self.split_into_heads(v)

        attention_weight = self.softmax(torch.einsum("bhqd,bhkd->bhqk", q_heads, k_heads) / self.head_dim ** 0.5)

        attention_matrix = torch.einsum("bhqk,bhvd->bhqd", attention_weight, v_heads)

        output = self.concat(attention_matrix)

        return output


class TransformerEncoder(nn.Module):
    def __init__(self, dim, n_heads, mlp_dim, depth):
        super(TransformerEncoder, self).__init__()

        self.norm = nn.LayerNorm(dim)
        self.mha = MultiHeadAttention(dim=dim, n_heads=n_heads)
        self.mlp = MLP(dim=dim, hidden_dim=mlp_dim)
        self.depth = depth

    def forward(self, x):
        for _ in range(self.depth):
            x = self.mha(self.norm(x)) + x
            x = self.mlp(self.norm(x)) + x

        return x

class MLPHead(nn.Module):
    def __init__(self, dim, out_dim):
        super(MLPHead, self).__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim)
        )
    
    def forward(self, x):
        x = self.net(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, n_classes, dim, depth, n_heads, channels=3, mlp_dim=256):
        super(VisionTransformer, self).__init__()

        # Params
        n_patches = (img_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.depth = depth

        # Layers
        self.patching = Patch(patch_size=patch_size)
        self.linear_projection_of_flattened_patches = LinearProjection(patch_dim=patch_dim, dim=dim)
        self.embedding = Embedding(n_patches=n_patches, dim=dim)
        self.transformer_encoder = TransformerEncoder(dim=dim, n_heads=n_heads, mlp_dim=mlp_dim, depth=depth)
        self.mlp_head = MLPHead(dim=dim, out_dim=n_classes)

    def forward(self, img):
        x = img

        """ 1. Patching
        x.shape: [batch_size, channels, img_height, img_width] -> [batch_size, n_patches, channels * (patch_size ** 2)]
        """
        x = self.patching(x)

        """ 2. Projection
        x.shape: [batch_size, n_patches, channels * (patch_size ** 2)] -> [batch_size, n_patches, dim]
        """
        x = self.linear_projection_of_flattened_patches(x)

        """ 3. Add cls token + Positional Encoding
        x.shape: [batch_size, n_patches, dim] -> [batch_size, n_patches + 1, dim]
        """
        x = self.embedding(x)

        """ 4. Transformer Encoder
        x.shape: No change
        """
        x = self.transformer_encoder(x)

        """ 5. MLP Head
        x.shape: [batch_size, n_patches + 1, dim] -> [batch_size, dim] -> [batch_size, n_classes]
        """
        # Take the first token (cls token)
        x = x[:, 0]
        x = self.mlp_head(x)

        return x