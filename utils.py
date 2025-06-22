import numpy as np
import torch
from quantus import denormalise
import torch.nn as nn
from config import device
import torch.nn.functional as F
import yaml
from skimage.util import random_noise


def minmax_norm(tensor):
    range_ = tensor.max() - tensor.min()
    if range_ != 0:
        tensor_norm = (tensor - tensor.min()) / range_
    else:
        tensor_norm = torch.ones_like(tensor) / len(tensor)

    return tensor_norm


def normalize_image(arr, dataset_name) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        arr_copy = arr.clone().cpu().numpy()
    else:
        arr_copy = arr.copy()

    if dataset_name == 'imagenet':
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        arr_copy = denormalise(arr_copy, mean=mean, std=std)
        arr_copy = np.moveaxis(arr_copy, 0, -1)

    elif dataset_name == 'mnist':
        arr_copy = np.reshape(arr_copy, (28, 28))

    elif dataset_name == 'fashionmnist':
        arr_copy = arr_copy / 2 + 0.5  # unnormalize
        arr_copy = np.reshape(arr_copy, (28, 28))

    elif dataset_name == 'cifar10':
        arr_copy = np.clip(arr_copy, 0, 1)
        arr_copy = np.moveaxis(arr_copy, 0, -1)

    else:
        assert False, f'Unknown dataset name: {dataset_name}'

    arr_copy = (arr_copy * 255.).astype(np.uint8)

    return arr_copy


def perturb_func(x, mask, mode):
    if mode == 'mask':
        return x * mask
    elif mode == 'interpolation':
        noise = torch.rand_like(x).to(device)
        # noise = torch.mean(x)
        x = x * mask + noise * (1 - mask)
        return x
    else:
        assert False, f'Unknown perturb_func {mode}'


def resampling(dataset, sample_weights, resample_size):
    if torch.sum(sample_weights) == 0:
        return dataset[[0]]

    resample_indices = torch.multinomial(sample_weights, num_samples=resample_size, replacement=True)

    samples = dataset[resample_indices]

    return samples


def sparsity(tensor, threshold=0.01):
    spars = torch.sum(torch.abs(tensor) < threshold) / tensor.numel()
    print(f"Sparsity(threshold={threshold}): {spars:.4f}")


class GaussianKernel(nn.Module):
    """
    https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/kde.py
    Implementation of the Gaussian kernel."""

    def __init__(self, return_log_kernel_value=False):
        super().__init__()
        # self.bandwidths = bandwidths
        self.return_log_kernel_value = return_log_kernel_value  # 计算密度的时候用到

    def _dist_mat(self, test_Xs, train_Xs, batch_size):
        n_test = test_Xs.shape[0]
        n_train = train_Xs.shape[0]
        assert not torch.isnan(train_Xs).any()
        assert not torch.isnan(test_Xs).any()

        dist_mat = torch.zeros(n_test, n_train, device=device) * -1

        for i in range(0, n_test, batch_size):
            start_idx_test = i
            end_idx_test = min(i + batch_size, n_test)

            for j in range(0, n_train, batch_size):
                start_idx_train = j
                end_idx_train = min(j + batch_size, n_train)

                test_batch = test_Xs[start_idx_test:end_idx_test]
                train_batch = train_Xs[start_idx_train:end_idx_train]

                diffs = test_batch.unsqueeze(1) - train_batch.unsqueeze(0)
                dist_batch = torch.norm(diffs, p=2, dim=-1)

                dist_mat[start_idx_test:end_idx_test, start_idx_train:end_idx_train] = dist_batch

        return dist_mat  # (n_test, n_train)

    def forward(self, bandwidths, test_Xs, train_Xs, feat_mask=None):
        num_channel = test_Xs.shape[1]
        if test_Xs.ndim != 2:
            batch_size = test_Xs.shape[0]
            test_Xs = test_Xs.reshape(batch_size, -1)
        if train_Xs.ndim != 2:
            batch_size = train_Xs.shape[0]
            train_Xs = train_Xs.reshape(batch_size, -1)
        if feat_mask is not None:
            if feat_mask.ndim != 2:
                if num_channel == 3:
                    feat_mask = feat_mask.expand(3, -1, -1)
                feat_mask = feat_mask.reshape(-1)

        n, d = train_Xs.shape
        n, h = torch.tensor(n, dtype=torch.float32), bandwidths.unsqueeze(0)
        pi = torch.pi

        test_Xs = test_Xs / h
        train_Xs = train_Xs / h

        if feat_mask is not None:
            # assert not torch.isnan(feat_mask).any()
            if torch.isnan(feat_mask).any():
                print(feat_mask)
                raise
                nan_indices = torch.isnan(feat_mask)
                feat_mask[nan_indices] = 0.

            feat_mask = feat_mask.unsqueeze(0)
            test_Xs = test_Xs * feat_mask
            train_Xs = train_Xs * feat_mask

        dist_mat = self._dist_mat(test_Xs, train_Xs, batch_size=64 * 2)  # (n_test, n_train)

        log_exp = -0.5 * (dist_mat ** 2)

        # return torch.logsumexp(log_exp - Z, dim=-1)

        if self.return_log_kernel_value:
            return log_exp
        else:
            return torch.exp(log_exp)


class AitchisonAitkenKernel(nn.Module):
    def __init__(self, num_levels):
        super().__init__()
        self.num_levels = num_levels

    def _dist_mat(self, bandwidths, test_Xs, train_Xs, batch_size, inv):
        n_test = test_Xs.shape[0]
        n_train = train_Xs.shape[0]
        n_feat = train_Xs.shape[1]
        assert not torch.isnan(train_Xs).any()
        assert not torch.isnan(test_Xs).any()

        dist_mat = torch.zeros(n_test, n_train, device=device) * -1

        for i in range(0, n_test, batch_size):
            start_idx_test = i
            end_idx_test = min(i + batch_size, n_test)

            for j in range(0, n_train, batch_size):
                start_idx_train = j
                end_idx_train = min(j + batch_size, n_train)

                test_batch = test_Xs[start_idx_test:end_idx_test]
                train_batch = train_Xs[start_idx_train:end_idx_train]

                diffs = test_batch.unsqueeze(1) - train_batch.unsqueeze(0)

                equal_mat = torch.eq(diffs, 0).float().to(device)
                if inv:

                    true_mat = (bandwidths / (self.num_levels - 1)).unsqueeze(0).to(device)
                    false_mat = (1 - bandwidths).unsqueeze(0).to(device)
                else:
                    true_mat = (1 - bandwidths).unsqueeze(0).to(device)
                    false_mat = (bandwidths / (self.num_levels - 1)).unsqueeze(0).to(device)
                diff_mat = equal_mat * true_mat + (1 - equal_mat) * false_mat  # (batch_test, batch_train, n_feat))

                log_dist_batch = torch.log(diff_mat).sum(dim=-1) - torch.log(bandwidths).sum(dim=-1)

                if n_feat > 50:
                    max_log = torch.max(log_dist_batch, dim=-1, keepdim=True).values
                    dist_batch = max_log * torch.exp(log_dist_batch - max_log)

                else:
                    dist_batch = torch.exp(log_dist_batch)

                dist_mat[start_idx_test:end_idx_test, start_idx_train:end_idx_train] = dist_batch

        return dist_mat  # (n_test, n_train)

    def forward(self, bandwidths, test_Xs, train_Xs, edge_mask=None, inv=False):
        h = bandwidths.unsqueeze(0)

        assert (h > 0).all()

        if edge_mask is not None:
            edge_mask = edge_mask.unsqueeze(0)
            test_Xs = test_Xs * edge_mask
            train_Xs = train_Xs * edge_mask

        dist_mat = self._dist_mat(bandwidths, test_Xs, train_Xs, batch_size=64 * 2, inv=inv)

        kernel_value = dist_mat

        return kernel_value


class MultivariateKDE(nn.Module):
    def __init__(self, train_data, feature_is_continuous, num_levels=None):
        super().__init__()
        # self.train_data = torch.tensor(np.array(train_data))
        self.train_data = train_data
        n, d = self.train_data.shape
        self.n = torch.tensor(n)
        self.d = torch.tensor(d)
        self.bw = self._compute_bw()
        if feature_is_continuous:
            self.kernel_func = GaussianKernel(return_log_kernel_value=True)
        else:
            assert num_levels is not None
            self.kernel_func = AitchisonAitkenKernel(num_levels)

        self.feature_is_continuous = feature_is_continuous

    def _compute_bw(self):
        """
        Returns Scott's normal reference rule of thumb bandwidth parameter.

        Notes
        -----
        See p.13 in [2] for an example and discussion.  The formula for the
        bandwidth is

        .. math:: h = 1.06n^{-1/(4+q)}

        where ``n`` is the number of observations and ``q`` is the number of
        variables.
        """

        X = torch.std(self.train_data, dim=0)
        X = torch.clamp(X, min=0.01)

        # return 1.06 * X * self.nobs ** (- 1. / (4 + self.train_data.shape[1]))
        bandwidths = 1.06 * X * self.n ** (- 1. / (4 + self.train_data.shape[1]))
        bandwidths = torch.clamp(bandwidths, max=0.49)

        return bandwidths

    def forward(self, test_data):
        kernel_value = self.kernel_func(self.bw, test_data, self.train_data)  # (n_test, n_train)

        if self.feature_is_continuous:
            log_kernel_value = kernel_value

            assert not torch.isnan(kernel_value).any()

            Z = 0.5 * self.d * torch.log(2 * torch.tensor(torch.pi)) + torch.log(self.bw).sum() + torch.log(self.n)

            diff = log_kernel_value - Z

            temperature = Z
            density = torch.sum(torch.exp(diff / temperature), dim=-1)

        else:
            density = torch.mean(kernel_value, dim=-1)

        return density


def compute_class_bandwidths(x_batch, model):
    logits = model(x_batch)
    num_classes = logits.shape[1]
    pred_class = torch.argmax(logits, dim=1)
    pred_class = F.one_hot(pred_class, num_classes).float()  # (n_samples, n_classes)
    kde = MultivariateKDE(pred_class, feature_is_continuous=False, num_levels=2)
    # pred_denisties = kde(pred_class)  # (n_samples,)
    return kde.bw  # (n_feat,)


def build_sample_data(target_x, size, mode='pepper', seed=1):
    target_x = target_x[0].cpu().numpy()

    samples = [target_x]
    for _ in range(size - 1):

        if mode == 'gaussian':
            x2 = random_noise(target_x, mode=mode, rng=seed, clip=True, mean=0, var=0.00001)
        else:
            x2 = random_noise(target_x, mode=mode, rng=seed, clip=True)

        samples.append(x2)

    samples = np.array(samples)
    samples = torch.from_numpy(samples).float().to(device)

    kde = MultivariateKDE(samples.reshape(size, -1), feature_is_continuous=True, num_levels=2)
    sample_denisties = kde(samples)
    feature_bandwidths = kde.bw  # (n_feat,)

    sample_data = [samples, sample_denisties]

    return sample_data, feature_bandwidths


def read_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    return data


def get_temperature(epoch: int, temp_params, total_epochs):
    """PG-Explainer: p14
    """
    return temp_params[0] * pow(temp_params[1] / temp_params[0], epoch / total_epochs)


def concrete_sample(logits, temperature=1.0, bias=0., scale=1.):
    eps = (1 - 2 * bias) * torch.rand_like(logits) + bias
    eps[eps == 0.] = 1e-6
    gumbels = eps.log() - (1 - eps).log()

    gumbels = gumbels * scale

    return (gumbels + logits) / temperature
