# Copyright Lars Andersen Bratholm 2024
"""
Variant of the multivariate TPE sampler that supports different parameters being sampled between trials,
without resorting to group decomposition.

This is done by allowing nan values for unsampled parameters in the internal representation. In estimating the
likelihoods, the univariate distributions in the product distributions are multiplied with a categorical
distribution indicating if the parameter is sampled or not. Except for categorical distributions where "nan"
is added as an additional choice.
"""

import copy
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import optuna
from numpy.typing import NDArray
from optuna.distributions import BaseDistribution, CategoricalDistribution
from optuna.samplers import TPESampler
from optuna.samplers._tpe import _truncnorm
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator, _ParzenEstimatorParameters
from optuna.samplers._tpe.probability_distributions import (
    _BatchedCategoricalDistributions,
    _BatchedDiscreteTruncNormDistributions,
    _BatchedDistributions,
    _BatchedTruncNormDistributions,
)
from optuna.samplers._tpe.sampler import _calculate_weights_below_for_multi_objective, _split_trials
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState


## search spaces
def _calculate_union(
    trials: list[optuna.trial.FrozenTrial],
    include_pruned: bool = False,
    search_space: Dict[str, BaseDistribution] | None = None,
    cached_trial_number: int = -1,
) -> Tuple[Dict[str, BaseDistribution] | None, int]:
    states_of_interest = [
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.WAITING,
        optuna.trial.TrialState.RUNNING,
    ]

    if include_pruned:
        states_of_interest.append(optuna.trial.TrialState.PRUNED)

    trials_of_interest = [trial for trial in trials if trial.state in states_of_interest]

    next_cached_trial_number = trials_of_interest[-1].number + 1 if len(trials_of_interest) > 0 else -1
    for trial in reversed(trials_of_interest):
        if cached_trial_number > trial.number:
            break

        if not trial.state.is_finished():
            next_cached_trial_number = trial.number
            continue

        if search_space is None:
            search_space = copy.copy(trial.distributions)
            continue

        search_space = {
            name: distribution
            for name, distribution in search_space.items()
            if name not in trial.distributions or trial.distributions.get(name) == distribution
        } | {
            name: distribution
            for name, distribution in trial.distributions.items()
            if name not in search_space or search_space.get(name) == distribution
        }

    return search_space, next_cached_trial_number


class UnionSearchSpace:
    """
    A class to calculate the union search space of a :class:`~optuna.study.Study`.

    Union search space contains the union of parameter distributions that have been
    suggested in the completed trials of the study so far.
    If there are multiple parameters that have the same name but different distributions,
    neither is included in the resulting search space
    (i.e., the parameters with dynamic value ranges are excluded).

    Note that an instance of this class is supposed to be used for only one study.

    Args:
        include_pruned:
            Whether pruned trials should be included in the search space.
    """

    def __init__(self, include_pruned: bool = False) -> None:
        self._cached_trial_number: int = -1
        self._search_space: Dict[str, BaseDistribution] | None = None
        self._study_id: int | None = None

        self._include_pruned = include_pruned

    def calculate(self, study: Study) -> Dict[str, BaseDistribution]:
        """
        Returns the Union search space of the :class:`~optuna.study.Study`.

        Args:
            study:
                A study with completed trials. The same study must be passed for one instance
                of this class through its lifetime.

        Returns:
            A dictionary containing the parameter names and parameter's distributions sorted by
            parameter names.
        """

        if self._study_id is None:
            self._study_id = study._study_id
        else:
            # Note that the check below is meaningless when `InMemoryStorage` is used
            # because `InMemoryStorage.create_new_study` always returns the same study ID.
            if self._study_id != study._study_id:
                raise ValueError("`UnionSearchSpace` cannot handle multiple studies.")

        self._search_space, self._cached_trial_number = _calculate_union(
            study.get_trials(deepcopy=False),
            self._include_pruned,
            self._search_space,
            self._cached_trial_number,
        )
        search_space = self._search_space or {}
        search_space = dict(sorted(search_space.items(), key=lambda x: x[0]))
        return copy.deepcopy(search_space)


# sampler
class NonGroupMultivariateTPESampler(TPESampler):
    """
    Variant of multivariate TPE that allows for trials with different sampled parameters, without using group
    decomposition.
    """

    def __init__(  # type: ignore
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, multivariate=True, group=False, warn_independent_sampling=False, **kwargs)
        self._search_space = UnionSearchSpace(include_pruned=True)  # type: ignore
        self._tpe_sampler = TPESampler(*args, multivariate=False, group=False, **kwargs)

    def _get_internal_repr(
        self, trials: list[FrozenTrial], search_space: dict[str, BaseDistribution]
    ) -> Dict[str, NDArray[np.float_]]:
        values: dict[str, list[float]] = {param_name: [] for param_name in search_space}
        for trial in trials:
            # Use nan as dummy values if not all parameters are sampled
            for param_name in search_space:
                if param_name in trial.params:
                    param = trial.params[param_name]
                    distribution = trial.distributions[param_name]
                    values[param_name].append(distribution.to_internal_repr(param))
                else:
                    values[param_name].append(np.nan)
        return {k: np.asarray(v) for k, v in values.items()}

    def _sample(
        self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
    ) -> Dict[str, Any]:
        if self._constant_liar and not study._is_multi_objective():
            states = [TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING]
        else:
            states = [TrialState.COMPLETE, TrialState.PRUNED]
        use_cache = not self._constant_liar
        trials = study._get_trials(deepcopy=False, states=states, use_cache=use_cache)

        # We divide data into below and above.
        n = sum(trial.state != TrialState.RUNNING for trial in trials)  # Ignore running trials.
        below_trials, above_trials = _split_trials(
            study,
            trials,
            self._gamma(n),
            self._constraints_func is not None,
        )

        below = self._get_internal_repr(below_trials, search_space)
        above = self._get_internal_repr(above_trials, search_space)

        # We then sample by maximizing log likelihood ratio.
        if study._is_multi_objective():
            weights_below = _calculate_weights_below_for_multi_objective(
                study, below_trials, self._constraints_func
            )
            mpe_below = _ParzenEstimator(
                below, search_space, self._parzen_estimator_parameters, weights_below
            )
        else:
            mpe_below = ParzenEstimator(below, search_space, self._parzen_estimator_parameters)
        mpe_above = ParzenEstimator(above, search_space, self._parzen_estimator_parameters)

        samples_below = mpe_below.sample(self._rng.rng, self._n_ei_candidates)
        log_likelihoods_below = mpe_below.log_pdf(samples_below)
        log_likelihoods_above = mpe_above.log_pdf(samples_below)
        ret = TPESampler._compare(samples_below, log_likelihoods_below, log_likelihoods_above)

        for param_name, dist in search_space.items():
            if np.isnan(ret[param_name]):
                # Fill with values from univariate tpe sampler if unsampled
                ret[param_name] = self._tpe_sampler.sample_independent(study, trial, param_name, dist)
            else:
                ret[param_name] = dist.to_external_repr(ret[param_name])

        return ret


# parzen_estimator
class ParzenEstimator(_ParzenEstimator):
    """
    Adds support for non sampled parameters.
    """

    def __init__(
        self,
        observations: Dict[str, NDArray[np.float_]],
        search_space: Dict[str, BaseDistribution],
        parameters: _ParzenEstimatorParameters,
        predetermined_weights: Optional[NDArray[np.float_]] = None,
    ) -> None:
        if parameters.consider_prior:
            if parameters.prior_weight is None:
                raise ValueError("Prior weight must be specified when consider_prior==True.")
            if parameters.prior_weight <= 0:
                raise ValueError("Prior weight must be positive.")

        self._search_space = search_space

        transformed_observations = self._transform(observations)

        assert predetermined_weights is None or len(transformed_observations) == len(predetermined_weights)
        weights = (
            predetermined_weights
            if predetermined_weights is not None
            else self._call_weights_func(parameters.weights, len(transformed_observations))
        )

        assert parameters.prior_weight is not None
        if len(transformed_observations) == 0:
            weights = np.array([1.0])
        elif parameters.consider_prior:
            weights = np.append(weights, [parameters.prior_weight])
        weights /= weights.sum()
        self._mixture_distribution = _MixtureOfProductDistribution(  # type: ignore
            consider_prior=parameters.consider_prior,
            prior_weight=parameters.prior_weight,
            weights=weights,
            distributions=[
                self._calculate_distributions(
                    transformed_observations[:, i], param, search_space[param], parameters
                )
                for i, param in enumerate(search_space)
            ],
        )

    def _calculate_categorical_distributions(
        self,
        observations: NDArray[np.float_],
        param_name: str,
        search_space: CategoricalDistribution,
        parameters: _ParzenEstimatorParameters,
    ) -> _BatchedDistributions:
        consider_prior = parameters.consider_prior or len(observations) == 0
        # Treat nan as a separate category
        assert parameters.prior_weight is not None
        weights = np.full(
            shape=(len(observations) + consider_prior, len(search_space.choices) + 1),
            fill_value=parameters.prior_weight / (len(observations) + consider_prior),
        )
        observations_ = observations.copy()
        observations_[np.isnan(observations)] = len(search_space.choices)
        observed_indices = observations_.astype(int)
        assert (observed_indices >= 0).all()
        assert (observed_indices <= len(search_space.choices)).all()

        weights[np.arange(len(observed_indices)), observed_indices] += 1
        weights /= weights.sum(axis=1, keepdims=True)
        return _BatchedCategoricalDistributions(weights)


class _MixtureOfProductDistribution(NamedTuple):
    weights: NDArray[np.float_]
    distributions: List[_BatchedDistributions]
    consider_prior: bool
    prior_weight: float

    def sample(self, rng: np.random.RandomState, batch_size: int) -> NDArray[np.float_]:
        active_indices = rng.choice(len(self.weights), p=self.weights, size=batch_size)

        ret = np.empty((batch_size, len(self.distributions)), dtype=np.float64)
        for i, d in enumerate(self.distributions):
            if isinstance(d, _BatchedCategoricalDistributions):
                active_weights = d.weights[active_indices, :]
                rnd_quantile = rng.rand(batch_size)
                cum_probs = np.cumsum(active_weights, axis=-1)
                assert np.isclose(cum_probs[:, -1], 1).all()
                cum_probs[:, -1] = 1  # Avoid numerical errors.
                ret[:, i] = np.sum(cum_probs < rnd_quantile[:, None], axis=-1)
                # Convert dummy choice back to nan
                ret[ret[:, i].astype(np.int64) == d.weights.shape[1] - 1, i] = np.nan
            elif isinstance(d, _BatchedTruncNormDistributions):
                active_mus = d.mu[active_indices]
                mask = ~np.isnan(active_mus)
                if sum(mask > 0):
                    active_sigmas = d.sigma[active_indices][mask]
                    ret[mask, i] = _truncnorm.rvs(
                        a=(d.low - active_mus[mask]) / active_sigmas,
                        b=(d.high - active_mus[mask]) / active_sigmas,
                        loc=active_mus[mask],
                        scale=active_sigmas,
                        random_state=rng,
                    )
                if sum(~mask) > 0:
                    ret[~mask, i] = np.nan
            elif isinstance(d, _BatchedDiscreteTruncNormDistributions):
                active_mus = d.mu[active_indices]
                mask = ~np.isnan(active_mus)
                if sum(mask) > 0:
                    active_sigmas = d.sigma[active_indices][mask]
                    samples = _truncnorm.rvs(
                        a=(d.low - d.step / 2 - active_mus[mask]) / active_sigmas,
                        b=(d.high + d.step / 2 - active_mus[mask]) / active_sigmas,
                        loc=active_mus[mask],
                        scale=active_sigmas,
                        random_state=rng,
                    )
                    ret[mask, i] = np.clip(
                        d.low + np.round((samples - d.low) / d.step) * d.step, d.low, d.high
                    )
                if sum(~mask) > 0:
                    ret[~mask, i] = np.nan
            else:
                assert False

        return ret

    def log_pdf(self, x: NDArray[np.float_]) -> NDArray[np.float_]:
        batch_size, n_vars = x.shape
        log_pdfs = np.zeros((batch_size, len(self.weights), n_vars), dtype=np.float64)
        for i, d in enumerate(self.distributions):
            xi = x[:, i]
            xi_mask = ~np.isnan(xi)
            if isinstance(d, _BatchedCategoricalDistributions):
                xi_ = xi.copy()
                xi_[~xi_mask] = d.weights.shape[1] - 1
                log_pdfs[:, :, i] = np.log(
                    np.take_along_axis(d.weights[None, :, :], xi_[:, None, None].astype(np.int64), axis=-1)
                )[:, :, 0]
            elif isinstance(d, _BatchedTruncNormDistributions):
                d_mu_mask = ~np.isnan(d.mu)
                p_nan = np.full(
                    shape=(d.mu.shape[0], 2),
                    fill_value=self.prior_weight / d.mu.shape[0],
                )
                p_nan[d_mu_mask, 0] += 1
                p_nan[~d_mu_mask, 1] += 1
                p_nan /= p_nan.sum(axis=1, keepdims=True)

                # Sample is not nan
                if sum(xi_mask) > 0:
                    if sum(d_mu_mask) > 0:
                        log_pdfs[np.ix_(xi_mask, d_mu_mask, np.arange(n_vars) == i)] = _truncnorm.logpdf(
                            x=xi[xi_mask, None],
                            a=(d.low - d.mu[None, d_mu_mask]) / d.sigma[None, d_mu_mask],
                            b=(d.high - d.mu[None, d_mu_mask]) / d.sigma[None, d_mu_mask],
                            loc=d.mu[None, d_mu_mask],
                            scale=d.sigma[None, d_mu_mask],
                        )[:, :, None]
                    if sum(~d_mu_mask) > 0:
                        # uniform when sample is not nan, while trial is nan
                        log_pdfs[np.ix_(xi_mask, ~d_mu_mask, np.arange(n_vars) == i)] -= np.log(
                            d.high - d.low
                        )
                    log_pdfs[xi_mask, :, i] += np.log(p_nan)[None, :, 0]
                # Sample is nan
                if sum(~xi_mask) > 0:
                    log_pdfs[~xi_mask, :, i] = np.log(p_nan)[None, :, 1]
            elif isinstance(d, _BatchedDiscreteTruncNormDistributions):
                lower_limit = d.low - d.step / 2
                upper_limit = d.high + d.step / 2

                d_mu_mask = ~np.isnan(d.mu)
                p_nan = np.full(
                    shape=(d.mu.shape[0], 2),
                    fill_value=self.prior_weight / d.mu.shape[0],
                )
                p_nan[d_mu_mask, 0] += 1
                p_nan[~d_mu_mask, 1] += 1
                p_nan /= p_nan.sum(axis=1, keepdims=True)

                # Sample is not nan
                if sum(xi_mask) > 0:
                    if sum(d_mu_mask) > 0:
                        x_lower = np.maximum(xi[xi_mask] - d.step / 2, lower_limit)
                        x_upper = np.minimum(xi[xi_mask] + d.step / 2, upper_limit)
                        log_gauss_mass = _truncnorm._log_gauss_mass(
                            (x_lower[:, None] - d.mu[None, d_mu_mask]) / d.sigma[None, d_mu_mask],
                            (x_upper[:, None] - d.mu[None, d_mu_mask]) / d.sigma[None, d_mu_mask],
                        )
                        log_p_accept = _truncnorm._log_gauss_mass(
                            (d.low - d.step / 2 - d.mu[None, d_mu_mask]) / d.sigma[None, d_mu_mask],
                            (d.high + d.step / 2 - d.mu[None, d_mu_mask]) / d.sigma[None, d_mu_mask],
                        )
                        log_pdfs[np.ix_(xi_mask, d_mu_mask, np.arange(n_vars) == i)] = (
                            log_gauss_mass - log_p_accept
                        )
                    if sum(~d_mu_mask) > 0:
                        # uniform when sample is not nan, while trial is nan
                        log_pdfs[np.ix_(xi_mask, ~d_mu_mask, np.arange(n_vars) == i)] += np.log(
                            d.step
                        ) - np.log(d.high - d.low + d.step)
            else:
                assert False

        weighted_log_pdf = np.sum(log_pdfs, axis=-1) + np.log(self.weights[None, :])
        max_ = weighted_log_pdf.max(axis=1)
        # We need to avoid (-inf) - (-inf) when the probability is zero.
        max_[np.isneginf(max_)] = 0
        with np.errstate(divide="ignore"):  # Suppress warning in log(0).
            per_sample_log_pdf: NDArray[np.float_] = (
                np.log(np.exp(weighted_log_pdf - max_[:, None]).sum(axis=1)) + max_
            )
            return per_sample_log_pdf
