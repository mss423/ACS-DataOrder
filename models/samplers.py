import math
import torch

def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()

def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()

def accuracy(ys_pred, ys):
    return (ys == ys_pred.sign()).float()

sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()

def get_gaussian_samples(n_dims, n_points, seed=None, scale=None, bias=None):
    if seed is None:
        xs = torch.randn(n_points, n_dims)
    else:
        generator = torch.Generator()
        generator.manual_seed(seed)
        xs = torch.randn(n_points, n_dims)

    if self.scale is not None:
        xs = xs @ scale
    if self.bias is not None:
        xs += bias
    return xs

def get_random_function(task, n_dims):
    if task == "linear":

    else:
        rprint("Unknown task")
        raise NotImplementedError

class Task:
    def __init__(self, n_dims, pool_dict=None, seeds=None):
        self.n_dims = n_dims
        self.pool_dict = pool_dict
        self.seeds = seeds
        assert pool_dict is None or seeds is None

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError

class LinearRegression(Task):
    def __init__(self, n_dims, pool_dict=None, seed=None, scale=1):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(LinearRegression, self).__init__(n_dims, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            self.w = torch.randn(self.n_dims, 1)
        elif seed is not None:
            self.w = torch.zeros(self.n_dims, 1)
            generator = torch.Generator()
            generator.manual_seed(seed)
            self.w = torch.randn(self.n_dims, 1, generator=generator)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w = pool_dict["w"][indices]

    def evaluate(self, xs):
        w = self.w.to(xs.device)
        ys = self.scale * (xs @ w)[:, :, 0]
        return ys

    @staticmethod
    def generate_pool_dict(n_dims, **kwargs):  # ignore extra args
        return {"w": torch.randn(n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error