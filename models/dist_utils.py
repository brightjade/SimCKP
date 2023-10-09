from torch.nn.parallel import DistributedDataParallel


class DDP(DistributedDataParallel):
    # Distributed wrapper. Supports asynchronous evaluation and model saving
    def forward(self, *args, **kwargs):
        # DDP has a sync point on forward. No need to do this for eval. This allows us to have different batch sizes
        if self.training: return super().forward(*args, **kwargs)
        else:             return self.module(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        self.module.load_state_dict(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)
