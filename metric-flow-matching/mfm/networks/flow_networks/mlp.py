import torch
import torch.nn as nn

from mfm.networks.mlp_base import ACTIVATION_MAP


class VelocityNet(nn.Module):
    """Velocity network with optional FiLM conditioning.

    When class_conditioning=True, applies Feature-wise Linear Modulation (FiLM)
    to the first hidden layer activations based on a condition input.
    The condition can be categorical (embedded) or continuous.
    Output is always ambient_dim-dimensional (spatial velocity only).
    """

    def __init__(
        self,
        dim: int,
        hidden_dims: list,
        activation: str,
        batch_norm: bool = False,
        class_conditioning: bool = False,
        categorical: bool = True,
        num_categories: int = None,
        cond_dim: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.class_conditioning = class_conditioning
        self.categorical = categorical

        act_cls = ACTIVATION_MAP[activation]
        input_size = dim + 1  # [t, x_spatial]

        # First hidden layer (before FiLM)
        self.first_linear = nn.Linear(input_size, hidden_dims[0])
        self.first_bn = nn.BatchNorm1d(hidden_dims[0]) if batch_norm else None
        self.first_act = act_cls()

        # FiLM conditioning layers
        if class_conditioning:
            if categorical:
                assert num_categories is not None, "num_categories required for categorical conditioning"
                self.cond_embedding_dim = max(16, hidden_dims[0] // 4)
                self.cond_embedding = nn.Embedding(num_categories, self.cond_embedding_dim)
                film_input_dim = self.cond_embedding_dim
            else:
                film_input_dim = cond_dim

            film_hidden_dim = max(16, hidden_dims[0] // 4)
            self.film_dense_0 = nn.Linear(film_input_dim, film_hidden_dim)
            self.film_act = nn.LeakyReLU()
            self.film_dense_1 = nn.Linear(film_hidden_dim, 2 * hidden_dims[0])

        # Remaining hidden layers + output
        rest_layers = []
        for i in range(len(hidden_dims) - 1):
            rest_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if batch_norm:
                rest_layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            rest_layers.append(act_cls())
        rest_layers.append(nn.Linear(hidden_dims[-1], dim))
        self.rest = nn.Sequential(*rest_layers)

    def forward(self, t, x, cond=None):
        if t.dim() < 1 or t.shape[0] != x.shape[0]:
            t = t.repeat(x.shape[0])[:, None]
        if t.dim() < 2:
            t = t[:, None]

        h = self.first_linear(torch.cat([t, x], dim=-1))
        if self.first_bn is not None:
            h = self.first_bn(h)

        if self.class_conditioning and cond is not None:
            if self.categorical:
                cond_emb = self.cond_embedding(cond.long())
            else:
                cond_emb = cond.unsqueeze(-1) if cond.dim() < 2 else cond
            film_params = self.film_act(self.film_dense_0(cond_emb))
            film_params = self.film_dense_1(film_params)
            hidden_dim = h.shape[-1]
            gamma = film_params[:, :hidden_dim]
            beta = film_params[:, hidden_dim:]
            h = gamma * h + beta

        h = self.first_act(h)
        return self.rest(h)
