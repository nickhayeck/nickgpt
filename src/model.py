import torch
import torch.nn as nn
import torch.nn.functional as nnf


class CausalSelfAttn(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        *,
        num_heads: int,
        dropout: float,
        bias: bool,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        assert embedding_dim % num_heads == 0, (
            f"bro. {embedding_dim} % {num_heads} != 0"
        )
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.inp_proj = nn.Linear(
            embedding_dim,
            3 * embedding_dim,
            dtype=dtype,
            device=device,
            bias=bias,
        )
        self.out_proj = nn.Linear(
            embedding_dim,
            embedding_dim,
            dtype=dtype,
            device=device,
            bias=bias,
        )
        self.out_proj_dropout = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        causal scaled dot-product self-attention

        Parameters
        ----------
        x: torch.Tensor
            shape of (..., seqlen, embedding_dim)
        """
        batch_dim, seqlen, embedding_dim = x.shape[:-2], x.shape[-2], x.shape[-1]
        q, k, v = self.inp_proj(x).split(embedding_dim, dim=-1)
        # fmt: off
        # reshape into "multi-head" format
        shape = (*batch_dim, seqlen, self.num_heads, embedding_dim // self.num_heads)
        q = q.reshape(*shape).transpose(-2, -3)
        k = k.reshape(*shape).transpose(-2, -3)
        v = v.reshape(*shape).transpose(-2, -3)
        # run sdpa kernel, reshape back to "single-head" format
        # use causal mask if no override is specified
        attn = nnf.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0,
            attn_mask=mask,
            is_causal=(mask is None),
        )
        attn = attn.transpose(-2, -3).reshape(*batch_dim, seqlen, embedding_dim)
        # fmt: on
        # output projection
        return self.out_proj_dropout(self.out_proj(attn))


class Block(nn.Module):
    def __init__(
        self,
        *,
        embedding_dim: int,
        attn_heads: int,
        dropout: float = 0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        assert 0 <= dropout <= 1, dropout
        super().__init__()
        # meta
        self.dtype = dtype
        self.device = device
        self.embedding_dim = embedding_dim
        # layers
        self.norm_0 = nn.LayerNorm([embedding_dim], dtype=dtype, device=device)
        self.attn = CausalSelfAttn(
            embedding_dim,
            num_heads=attn_heads,
            dropout=dropout,
            bias=False,
            dtype=dtype,
            device=device,
        )
        # construct MLP
        _mlp = [
            nn.Linear(embedding_dim, 4 * embedding_dim, dtype=dtype, device=device),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim, dtype=dtype, device=device),
        ]
        if dropout > 0:
            _mlp.append(nn.Dropout(dropout))
        self.norm_1 = nn.LayerNorm([embedding_dim], dtype=dtype, device=device)
        self.mlp = nn.Sequential(*_mlp)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None):
        """
        Parameters
        ----------
        x: torch.Tensor
            shape of (batch_size, seqlen, self.embedding_dim)
        """
        assert x.shape[-1] == self.embedding_dim, (
            f"{x.shape[-1]} != {self.embedding_dim}"
        )
        x = x + self.attn(self.norm_0(x), mask=attn_mask)
        x = x + self.mlp(self.norm_1(x))
        return x


# TODO: custom init once pre-training pipeline is solid
class GPT(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        context_size: int,
        embedding_dim: int,
        attn_heads: int,
        num_blocks: int,
        dropout: float = 0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.device = device
        self.context_size = context_size
        self.vocab_embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            dtype=dtype,
            device=device,
        )
        self.pos_embedding = nn.Embedding(
            context_size,
            embedding_dim,
            dtype=dtype,
            device=device,
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    embedding_dim=embedding_dim,
                    attn_heads=attn_heads,
                    dropout=dropout,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(num_blocks)
            ]
        )
        self.norm = nn.LayerNorm([embedding_dim], device=device, dtype=dtype)

        # manually use the embedding weights with the linear function.
        # the weights are transposed because nn.Linear expects (out_dim, in_dim)
        # and nn.Embedding stores (vocab_size, embedding_dim)
        self.head = lambda x: nnf.linear(x, self.vocab_embedding.weight, bias=None)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        inference_mode: bool = False,
    ):
        """
        Parameters
        ----------
        x: torch.Tensor
            shape of (batch_size, seqlen)
        inference_mode: bool
            in the final output head, we can ignore all but the last coordinate
            in the "sequence" dimension, we return a smaller
        """
        seqlen = x.shape[1]
        assert seqlen <= self.context_size, f"{seqlen} > {self.context_size}"
        pos = self.pos_embedding(torch.arange(seqlen, device=x.device).long())

        x = pos + self.vocab_embedding(x)
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        x = self.norm(x)
        if inference_mode:
            # only calc the head for the last token
            return self.head(x[:, [-1], :])
        return self.head(x)
