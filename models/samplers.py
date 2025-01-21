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
        xs = torch.randn(n_dims, n_points)
    else:
        generator = torch.Generator()
        generator.manual_seed(seed)
        xs = torch.randn(n_dims, n_points)

    if scale is not None:
        xs = xs @ scale
    if bias is not None:
        xs += bias
    return xs

class Task:
    def __init__(self, n_dims, pool_dict=None, seed=None):
        self.n_dims = n_dims
        self.pool_dict = pool_dict
        self.seed = seed
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
        super(LinearRegression, self).__init__(n_dims, pool_dict, seed)
        self.scale = scale

        if pool_dict is None and seed is None:
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
        ys = self.scale * (xs.T @ w)
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

class SparseLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        pool_dict=None,
        seeds=None,
        scale=1,
        sparsity=3,
        valid_coords=None,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(SparseLinearRegression, self).__init__(
            n_dims, pool_dict, seeds, scale
        )
        self.sparsity = sparsity
        if valid_coords is None:
            valid_coords = n_dims
        assert valid_coords <= n_dims

        mask = torch.ones(n_dims).bool()
        if seeds is None:
            perm = torch.randperm(valid_coords)
        else:
            generator = torch.Generator()
            generator.manual_seed(seeds[i])
            perm = torch.randperm(valid_coords, generator=generator)
        mask[perm[:sparsity]] = False
        self.w[mask] = 0

    def evaluate(self, xs):
        w = self.w.to(xs.device)
        ys = self.scale * (xs.T @ w)
        return ys

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

class DecisionTree(Task):
    def __init__(self, n_dims, pool_dict=None, seeds=None, depth=4):

        super(DecisionTree, self).__init__(n_dims, pool_dict, seeds)
        self.depth = depth

        if pool_dict is None:

            # We represent the tree using an array (tensor). Root node is at index 0, its 2 children at index 1 and 2...
            # dt_tensor stores the coordinate used at each node of the decision tree.
            # Only indices corresponding to non-leaf nodes are relevant
            self.dt_tensor = torch.randint(
                low=0, high=n_dims, size=(1, 2 ** (depth + 1) - 1)
            )

            # Target value at the leaf nodes.
            # Only indices corresponding to leaf nodes are relevant.
            self.target_tensor = torch.randn(self.dt_tensor.shape)
        elif seeds is not None:
            self.dt_tensor = torch.zeros(1, 2 ** (depth + 1) - 1)
            self.target_tensor = torch.zeros_like(dt_tensor)
            generator = torch.Generator()
            generator.manual_seed(seed)
            self.dt_tensor[0] = torch.randint(
                low=0,
                high=n_dims - 1,
                size=2 ** (depth + 1) - 1,
                generator=generator,
            )
            self.target_tensor[0] = torch.randn(
                self.dt_tensor[0].shape, generator=generator
            )
        else:
            raise NotImplementedError

    def evaluate(self, xs):
        dt_tensor = self.dt_tensor.to(xs.device)
        target_tensor = self.target_tensor.to(xs.device)

        # Assume xs is a single data point
        xs_bool = xs > 0

        # Use the single decision tree
        dt = dt_tensor[0]  
        target = target_tensor[0] 

        cur_nodes = torch.zeros(xs.shape[0], device=xs.device).long() 
        for j in range(self.depth):
            cur_coords = dt[cur_nodes]
            cur_decisions = xs_bool[torch.arange(xs_bool.shape[0]), cur_coords]
            cur_nodes = 2 * cur_nodes + 1 + cur_decisions

        # Return the target value for the final node
        return target[cur_nodes]

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

