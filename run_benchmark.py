# Copyright Lars Andersen Bratholm 2024

import copy
import functools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import optuna
from hierarchical_tpe import HierarchicalTPESampler
from multivariate_tpe import NonGroupMultivariateTPESampler
from numpy.typing import NDArray
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import BaseSampler, RandomSampler, TPESampler
from optuna.trial import Trial
from tqdm.contrib.concurrent import process_map


def objective(trial: Trial) -> float:
    """
    Objective to minimize during hyper-parameter optimization.

    :param trial: the optuna trial
    :returns: loss
    """
    x = trial.suggest_categorical("x", [True, False])
    y = trial.suggest_float("y", -1, 1)
    if x is True:
        n = trial.suggest_categorical("n", [True, False])
        if n is True:
            a = trial.suggest_float("a", -1, 1)
            return (a - y) ** 2 + (a + 0.75) ** 2 + 0.025
        b = trial.suggest_float("b", -1, 1)
        return (b - y) ** 2 + (b + 0.25) ** 2 + 0.05
    else:
        m = trial.suggest_categorical("m", [True, False])
        if m is True:
            c = trial.suggest_float("c", -1, 1)
            return (c - y) ** 2 + (c - 0.25) ** 2 + 0.4
        d = trial.suggest_float("d", -1, 1)
        return (d - y) ** 2 + (d - 0.75) ** 2 + 0.01


def run_optimization(n_trials: int, sampler: BaseSampler) -> NDArray[np.float_]:
    sampler_ = sampler = copy.deepcopy(sampler)
    study = optuna.create_study(sampler=sampler_, direction="minimize")
    # Seed with all combinations of categorical parameters
    study.enqueue_trial({"x": True, "n": True})
    study.enqueue_trial({"x": True, "n": False})
    study.enqueue_trial({"x": False, "m": True})
    study.enqueue_trial({"x": False, "m": False})

    study.optimize(objective, n_trials=n_trials)
    # Shouldn't be any None values
    values = [trial.value for trial in study.trials if trial.value is not None]
    best_values = np.minimum.accumulate(values)
    return best_values


def run_benchmarks(n_trials: int, n_repetitions: int, max_workers: int) -> None:
    r1 = run_single_benchmark(n_trials, n_repetitions, RandomSampler(), max_workers)
    r2 = run_single_benchmark(
        n_trials,
        n_repetitions,
        TPESampler(multivariate=False, n_startup_trials=8, n_ei_candidates=32),
        max_workers,
    )
    r3 = run_single_benchmark(
        n_trials,
        n_repetitions,
        TPESampler(multivariate=True, group=True, n_startup_trials=8, n_ei_candidates=256),
        max_workers,
    )
    r4 = run_single_benchmark(
        n_trials,
        n_repetitions,
        NonGroupMultivariateTPESampler(n_startup_trials=8, n_ei_candidates=16),
        max_workers,
    )
    r5 = run_single_benchmark(
        n_trials,
        n_repetitions,
        HierarchicalTPESampler(n_startup_trials=8, n_ei_candidates=8),
        max_workers,
    )
    plt.plot(r1, label="random")
    plt.plot(r3, label="tpe")
    plt.plot(r2, label="multivariate tpe")
    plt.plot(r4, label="multivariate tpe (no group)")
    plt.plot(r5, label="hierarchical tpe")
    # plt.plot(r6, label="simple")
    plt.legend()
    plt.xlabel("Trials")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.savefig("benchmark.png", dpi=400)
    plt.clf()


def run_single_benchmark(
    n_trials: int, n_repetitions: int, sampler: BaseSampler, max_workers: int
) -> NDArray[np.float_]:
    results = process_map(
        functools.partial(run_optimization, sampler=sampler),
        (n_trials for _ in range(n_repetitions)),
        max_workers=max_workers,
        desc=sampler.__class__.__name__,
        total=n_repetitions,
    )
    # Take geometric mean at each step
    geometric_mean: NDArray[np.float_] = np.exp(np.sum(np.log(results), axis=0) / n_repetitions)
    return geometric_mean


if __name__ == "__main__":
    np.random.seed(42)
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    warnings.simplefilter("ignore", category=(UserWarning, ExperimentalWarning))  # type: ignore
    run_benchmarks(n_trials=500, n_repetitions=128, max_workers=5)
