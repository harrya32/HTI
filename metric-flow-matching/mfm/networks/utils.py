import torch


class flow_model_torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format.

    When cond is provided, it is passed as side information to the model
    at every ODE integration step. The condition stays fixed throughout
    the trajectory — only the spatial state x evolves.
    """

    def __init__(self, model, cond=None):
        super().__init__()
        self.model = model
        self.cond = cond

    def forward(self, t, x, *args, **kwargs):
        if self.cond is not None:
            return self.model(t, x, cond=self.cond)
        return self.model(t, x)
