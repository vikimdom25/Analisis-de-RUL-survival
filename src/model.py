"""
Modelo completo: encoder causal de dos etapas + hazard head + RUL head,
más la matemática de supervivencia (S(t) y la verosimilitud censurada).

Corre en CPU o GPU automáticamente (el dispositivo se decide en
train.py) -- pensado para subir a Colab con GPU sin cambios.

LIMITACIÓN CONOCIDA (aceptada deliberadamente, ver decisión del
proyecto): el encoder no tiene ventana de contexto acotada -- cada
h(t) atiende a TODO el historial desde el ciclo 1, la memoria escala
como O(batch × sensores × T²). Válido para el rango de vidas de FD001
(hasta 362 ciclos); el arreglo real (ventana deslizante acotada)
queda pendiente para una v2.0.
"""

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------- encoder causal

def causal_mask(T: int, device) -> torch.Tensor:
    """(T, T) bool: True donde NO se permite atender (futuro)."""
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)


class MultiHeadAttentionSDPA(nn.Module):
    """
    Usa F.scaled_dot_product_attention (kernels de flash-attention en
    GPU; en CPU cae al backend genérico -- ver nota de memoria arriba).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x, attn_mask=None, key_padding_mask=None, is_causal=False):
        B, T, dm = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        mask = None
        if key_padding_mask is not None:
            mask = ~key_padding_mask[:, None, None, :]
            if attn_mask is not None:
                mask = mask & ~attn_mask[None, None, :, :]
        elif attn_mask is not None:
            mask = ~attn_mask[None, None, :, :]

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=(is_causal and mask is None),
        )
        out = out.transpose(1, 2).reshape(B, T, dm)
        return self.out_proj(out), None


class SensorEmbedding(nn.Module):
    def __init__(self, n_sensors: int, d_model: int):
        super().__init__()
        self.value_proj = nn.Linear(1, d_model)
        self.channel_embed = nn.Parameter(torch.randn(n_sensors, d_model) * 0.02)

    def forward(self, x):  # x: (B, T, D) -> (B, T, D, d_model)
        return self.value_proj(x.unsqueeze(-1)) + self.channel_embed


class CausalTwoStageBlock(nn.Module):
    """Etapa 1: causal en el tiempo, por sensor. Etapa 2: entre sensores, por ciclo."""

    def __init__(self, d_model: int, n_heads: int = 4, ff_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.temporal_attn = MultiHeadAttentionSDPA(d_model, n_heads, dropout=dropout)
        self.sensor_attn = MultiHeadAttentionSDPA(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * ff_mult), nn.GELU(), nn.Linear(d_model * ff_mult, d_model))

    def forward(self, x, key_padding_mask=None):
        B, T, D, dm = x.shape

        x1 = x.permute(0, 2, 1, 3).reshape(B * D, T, dm)
        mask = causal_mask(T, x.device)
        kpm = key_padding_mask.unsqueeze(1).expand(B, D, T).reshape(B * D, T) if key_padding_mask is not None else None
        attn_out, _ = self.temporal_attn(x1, attn_mask=mask, key_padding_mask=kpm, is_causal=True)
        x1 = self.norm1(x1 + attn_out).reshape(B, D, T, dm).permute(0, 2, 1, 3)

        x2 = x1.reshape(B * T, D, dm)
        attn_out2, _ = self.sensor_attn(x2)
        x2 = self.norm2(x2 + attn_out2).reshape(B, T, D, dm)

        return self.norm3(x2 + self.ff(x2))


class CausalTwoStageEncoder(nn.Module):
    def __init__(self, n_sensors: int, d_model: int = 32, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.embed = SensorEmbedding(n_sensors, d_model)
        self.blocks = nn.ModuleList([CausalTwoStageBlock(d_model, n_heads) for _ in range(n_layers)])

    def forward(self, x, key_padding_mask=None):
        h = self.embed(x)
        for block in self.blocks:
            h = block(h, key_padding_mask=key_padding_mask)
        return h.mean(dim=2)  # pooling entre sensores -> (B, T, d_model)


# ---------------------------------------------------------------- cabezas de salida

class HazardHead(nn.Module):
    def __init__(self, d_model: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, h):
        return torch.sigmoid(self.net(h).squeeze(-1))


class RULHead(nn.Module):
    """Cabeza auxiliar de multitask mínimo: regresión directa de RUL."""

    def __init__(self, d_model: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, h):
        return self.net(h).squeeze(-1)


class CausalSurvivalModel(nn.Module):
    def __init__(self, n_sensors: int, d_model: int = 32, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.encoder = CausalTwoStageEncoder(n_sensors, d_model, n_layers, n_heads)
        self.hazard_head = HazardHead(d_model)
        self.rul_head = RULHead(d_model)

    def forward(self, x, key_padding_mask=None):
        embed = self.encoder(x, key_padding_mask=key_padding_mask)
        return self.hazard_head(embed), self.rul_head(embed)


# ---------------------------------------------------------------- matemática de supervivencia

def compute_log_survival(hazard: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """logS_cum[:, t] = log S(t) = suma acumulada de log(1 - h(k)), k=0..t."""
    h_clamped = hazard.clamp(eps, 1 - eps)
    return torch.cumsum(torch.log1p(-h_clamped), dim=1)


def censored_nll(hazard: torch.Tensor, event: torch.Tensor, lengths: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    log L_i = delta_i * [logS(T_i - 1) + log h(T_i)] + (1-delta_i) * logS(T_i)
    """
    logS_cum = compute_log_survival(hazard, eps=eps)
    B = hazard.size(0)
    idx = torch.arange(B, device=hazard.device)
    last_idx = (lengths - 1).clamp(min=0)

    logS_at_last = logS_cum[idx, last_idx]
    h_at_last = hazard[idx, last_idx].clamp(eps, 1 - eps)

    prev_idx = (last_idx - 1).clamp(min=0)
    logS_at_prev = torch.where(last_idx > 0, logS_cum[idx, prev_idx], torch.zeros_like(logS_at_last))

    log_lik = event * (logS_at_prev + torch.log(h_at_last)) + (1 - event) * logS_at_last
    return -log_lik.mean()
