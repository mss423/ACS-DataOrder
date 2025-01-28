import json
import os
import sys

from munch import Munch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import yaml

import models
from samplers import get_data_sampler, sample_transformation
from tasks import get_task_sampler
from order_methods import get_order


# Functions for evaluation


def eval_batch(model, task_sampler, xs, xs_p=None):
    task = task_sampler()
    if torch.cuda.is_available() and model.name.split("_")[0] in ["gpt2", "lstm"]:
        device = "cuda"
    else:
        device = "cpu"

    if xs_p is None:
        ys = task.evaluate(xs)
        pred = model(xs.to(device), ys.to(device)).detach()
        metrics = task.get_metric()(pred.cpu(), ys)
    else:
        b_size, n_points, _ = xs.shape
        metrics = torch.zeros(b_size, n_points)
        for i in range(n_points):
            xs_comb = torch.cat((xs[:, :i, :], xs_p[:, i:, :]), dim=1)
            ys = task.evaluate(xs_comb)

            pred = model(xs_comb.to(device), ys.to(device), inds=[i]).detach()
            metrics[:, i] = task.get_metric()(pred.cpu(), ys)[:, i] # test on next point in sequence

    return metrics

def eval_batch_random(model, task_sampler, xs, xs_p=None):
    task = task_sampler()
    if torch.cuda.is_available() and model.name.split("_")[0] in ["gpt2", "lstm"]:
        device = "cuda"
    else:
        device = "cpu"

    if xs_p is None:
        ys = task.evaluate(xs)
        pred, ids = model(xs.to(device), ys.to(device))#3.detach()
        metrics = task.get_metric()(pred.cpu(), ys[:,ids])
    else:
        b_size, n_points, _ = xs.shape
        metrics = torch.zeros(b_size, n_points)
        for i in range(n_points):
            xs_comb = torch.cat((xs[:, :i, :], xs_p[:, i:, :]), dim=1)
            ys = task.evaluate(xs_comb)

            pred, ids = model(xs_comb.to(device), ys.to(device), inds=[i])#.detach()
            metrics[:, i] = task.get_metric()(pred.cpu(), ys)[:, i] # test on next point in sequence

    return metrics


# Functions for generating different kinds of train/test data


def gen_standard(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)

    return xs, None


def gen_opposite_quadrants(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    pattern = torch.randn([b_size, 1, xs.shape[2]]).sign()

    xs_train_pre = xs.abs() * pattern
    xs_test_post = -xs_train_pre

    return xs_train_pre, xs_test_post


def gen_random_quadrants(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    pattern = torch.randn([b_size, 1, xs.shape[2]]).sign()

    xs_train_pre = xs.abs() * pattern
    xs_test_post = xs

    return xs_train_pre, xs_test_post


def gen_orthogonal_train_test(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    n_dim = xs.shape[2]
    n_points = min(n_points, n_dim)
    # raise ValueError("number of points should be at most the dimension.")
    xs_train_pre = xs
    xs_test_post = torch.zeros(xs.shape)
    for i in range(n_points):
        xs_test_post_i = xs[:, i : i + 1, :]
        xs_train_pre_i = xs[:, :i, :]
        _, _, Vt = torch.linalg.svd(xs_train_pre_i, full_matrices=False)
        xs_train_pre_i_projection = Vt.transpose(1, 2) @ Vt
        xs_test_post_i_orthogonalized = (
            xs_test_post_i - xs_test_post_i @ xs_train_pre_i_projection
        )
        xs_test_post_i_normalized = (
            xs_test_post_i_orthogonalized
            * xs_test_post_i.norm(dim=2).unsqueeze(2)
            / xs_test_post_i_orthogonalized.norm(dim=2).unsqueeze(2)
        )

        xs_test_post[:, i : i + 1, :] = xs_test_post_i_normalized

    return xs_train_pre, xs_test_post


def gen_overlapping_train_test(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    xs_train_pre = xs
    xs_test_post = xs.clone()
    b_size = xs.shape[0]
    for i in range(1, n_points):
        xs_train_pre_i = xs[:, :i, :]
        perm = torch.stack([torch.randperm(i) for _ in range(b_size)]).unsqueeze(dim=1)
        ind_mat = (perm == 0) + 0.0
        xs_test_post[:, i : i + 1, :] = ind_mat @ xs_train_pre_i

    return xs_train_pre, xs_test_post


def aggregate_metrics(metrics, bootstrap_trials=1000):
    """
    Takes as input a tensor of shape (num_eval, n_points) and returns a dict with
    per-point mean, stddev, and bootstrap limits
    """
    results = {}
    results["mean"] = metrics.mean(dim=0)
    results["std"] = metrics.std(dim=0, unbiased=True)
    n = len(metrics)
    bootstrap_indices = torch.randint(n, size=(bootstrap_trials, n))
    bootstrap_means = metrics[bootstrap_indices].mean(dim=1).sort(dim=0)[0]
    results["bootstrap_low"] = bootstrap_means[int(0.05 * bootstrap_trials), :]
    results["bootstrap_high"] = bootstrap_means[int(0.95 * bootstrap_trials), :]

    return {k: v.tolist() for k, v in results.items()}


def eval_model(
    model,
    task_name,
    n_dims,
    n_points,
    data_name="gaussian",
    prompting_strategy="standard",
    num_eval_examples=1280,
    batch_size=64,
    method=None, # method of ordering data points
    data_sampler_kwargs={},
    task_sampler_kwargs={},
):
    """
    Evaluate a model on a task with a variety of strategies.
       Args:
       - task: which base task we are evaluating on. E.g., "linear_regression"
       - prompting_strategy: how to construct the prompt, e.g., "random_quadrants"
       - num_eval_examples: total number of examples to evaluate on
       - **sampler_kwargs: remaining arguments to pass directly to the sampler
    """

    assert num_eval_examples % batch_size == 0
    data_sampler = get_data_sampler(data_name, n_dims, **data_sampler_kwargs)
    task_sampler = get_task_sampler(
        task_name, n_dims, batch_size, **task_sampler_kwargs
    )

    all_metrics = []

    generating_func = globals()[f"gen_{prompting_strategy}"]
    for i in range(num_eval_examples // batch_size):
        xs, xs_p = generating_func(data_sampler, n_points, batch_size)
        if method:
            order = get_order(xs, method)
            xs = xs[torch.arange(batch_size)[:, None, None], order, torch.arange(n_dims)]

        metrics = eval_batch(model, task_sampler, xs, xs_p)
        all_metrics.append(metrics)

    metrics = torch.cat(all_metrics, dim=0)

    return aggregate_metrics(metrics)

def eval_model_random(
    model,
    task_name,
    n_dims,
    n_points,
    data_name="gaussian",
    prompting_strategy="standard",
    num_eval_examples=1280,
    batch_size=64,
    method=None, # method of ordering data points
    threshold=0.5,
    data_sampler_kwargs={},
    task_sampler_kwargs={},
    **kwargs
):
    """
    Evaluate a model on a task with a variety of strategies.
       Args:
       - task: which base task we are evaluating on. E.g., "linear_regression"
       - prompting_strategy: how to construct the prompt, e.g., "random_quadrants"
       - num_eval_examples: total number of examples to evaluate on
       - **sampler_kwargs: remaining arguments to pass directly to the sampler
    """

    assert num_eval_examples % batch_size == 0
    data_sampler = get_data_sampler(data_name, n_dims, **data_sampler_kwargs)
    task_sampler = get_task_sampler(
        task_name, n_dims, batch_size, **task_sampler_kwargs
    )

    all_metrics = []

    generating_func = globals()[f"gen_{prompting_strategy}"]
    for i in tqdm(range(num_eval_examples // batch_size)):
        xs, xs_p = generating_func(data_sampler, n_points, batch_size)
        if method:
            order = get_order(xs, method, **kwargs)
            xs = xs[torch.arange(batch_size)[:, None, None], order, torch.arange(n_dims)]

        metrics = eval_batch_random(model, task_sampler, xs, xs_p)
        all_metrics.append(metrics)

    metrics = torch.cat(all_metrics, dim=0)

    return aggregate_metrics(metrics)


def collect_results(
    task_name, 
    models, 
    n_dims, 
    n_points, 
    order_methods=None
):
    
    results = {}
    for method in order_methods:
        print(f"Running with {method}")
        metrics = {}
        for model in models:
            if method == "random":
                metrics[model.name] = eval_model(model, task_name, n_dims, n_points, method=None)
            else:
                metrics[model.name] = eval_model(model, task_name, n_dims, n_points, method=method)
        results[method] = metrics
    return results

def collect_results_random(
    task_name, 
    models, 
    n_dims, 
    n_points, 
    order_methods=None, 
    batch_size=64, 
    num_eval_examples=1280,
    prompting_strategy="standard",
    sweep=False,
    **kwargs
):
    
    results = {}
    for method in order_methods:
        print(f"Running with {method}")
        metrics = {}
        for model in models:
            if method == "random":
                metrics[model.name] = eval_model_random(model, task_name, n_dims, n_points, 
                    method=None, 
                    batch_size=batch_size, 
                    num_eval_examples=num_eval_examples,
                    prompting_strategy=prompting_strategy,
                    **kwargs)
            else:
                metrics[model.name] = eval_model_random(model, task_name, n_dims, n_points, 
                    method=method, 
                    batch_size=batch_size, 
                    num_eval_examples=num_eval_examples,
                    prompting_strategy=prompting_strategy,
                    **kwargs)

        results[method] = metrics
    return results
