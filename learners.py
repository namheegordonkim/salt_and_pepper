from abc import abstractmethod

import numpy as np
import torch
import scipy

from torch import nn


class RewardFunction:
    def __init__(self, mu: np.ndarray, sigma: float):
        self.mu = mu
        self.sigma = sigma

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pdf_x = scipy.stats.norm.pdf(x, loc=self.mu[0], scale=self.sigma)
        pdf_y = scipy.stats.norm.pdf(y, loc=self.mu[1], scale=self.sigma)
        return pdf_x * pdf_y


class ContextualRewardFunction:
    def __init__(self, mus: np.ndarray, sigmas: np.ndarray):
        self.mus = mus
        self.sigmas = sigmas

    def evaluate(
        self, context_idx: np.ndarray, x: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        pdf_x = scipy.stats.norm.pdf(
            x, loc=self.mus[context_idx][..., 0], scale=self.sigmas[context_idx]
        )
        pdf_y = scipy.stats.norm.pdf(
            y, loc=self.mus[context_idx][..., 1], scale=self.sigmas[context_idx]
        )
        return pdf_x * pdf_y


class Learner:
    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def sample(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError

    @abstractmethod
    def weigh(self):
        raise NotImplementedError

    @abstractmethod
    def learn(self):
        raise NotImplementedError


class BaseLearner(Learner):
    def __init__(self, reward_function: RewardFunction):
        self.reward_function = reward_function

        self.samples = None
        self.rewards = None
        self.weights = None
        self.mu = None
        self.sigma = None

    def reset(self, init_sigma: float = 1.0):
        self.samples = np.empty((0, 2))
        self.rewards = np.empty((0,))
        self.weights = np.empty((0,))
        self.mu = np.zeros(2)
        self.sigma = np.ones(2) * init_sigma

    def sample(self, n_samples: int = 10):
        noise = np.random.normal(0, 1, (n_samples, 2))
        self.samples = self.mu + self.sigma * noise

        self.rewards = np.zeros(n_samples)
        self.weights = np.zeros(n_samples)

    def evaluate(self):
        self.rewards = self.reward_function.evaluate(
            self.samples[:, 0], self.samples[:, 1]
        )
        self.weights = np.zeros_like(self.rewards)

    def weigh(self, elite_proportion: float = 0.5):
        # Elite selection weighting
        reward_argsort = np.argsort(self.rewards)
        self.weights = np.zeros_like(self.rewards)
        self.weights[
            reward_argsort[-int(self.samples.shape[0] * elite_proportion) :]
        ] = 1

    @abstractmethod
    def learn(self):
        raise NotImplementedError


class CEMLearner(BaseLearner):

    def learn(self):
        """
        CEM: simply to weighted average based on 1s ans 0s
        """
        self.mu = np.sum(self.weights[:, None] * self.samples, axis=0) / np.sum(
            self.weights
        )
        var = np.sum(
            self.weights[:, None] * (self.samples - self.mu[None]) ** 2, axis=0
        ) / np.sum(self.weights)
        self.sigma = np.sqrt(var)
        self.sigma = np.maximum(self.sigma, 0.001)


class NLLLearner(BaseLearner):

    def learn(self, n_iter: int = 1000, lr: float = 0.01):
        """
        NLL: use negative log-likelihood to optimize mu and sigma
        """
        mu = nn.Parameter(torch.as_tensor(self.mu, dtype=torch.float))
        log_sigma = nn.Parameter(torch.as_tensor(np.log(self.sigma), dtype=torch.float))
        optimizer = torch.optim.Adam(params=[mu, log_sigma], lr=lr)
        weights = torch.as_tensor(self.weights, dtype=torch.float)
        for _ in range(n_iter):
            optimizer.zero_grad()

            dist = torch.distributions.Normal(mu, torch.exp(log_sigma))
            log_probs = dist.log_prob(
                torch.tensor(self.samples, dtype=torch.float)
            ).mean(dim=1)
            loss = -torch.mean(weights * log_probs)
            loss.backward()

            optimizer.step()
        self.mu = mu.cpu().detach().numpy()
        self.sigma = torch.exp(log_sigma).cpu().detach().numpy()


class ScaleShift(nn.Module):
    def __init__(self, size, scale, shift):
        super().__init__()
        self.size = size
        self.scale = nn.Parameter(torch.ones(size, dtype=torch.float) * scale)
        self.shift = nn.Parameter(torch.ones(size, dtype=torch.float) * shift)

    def forward(self, x):
        return x * self.scale + self.shift


class ContextualLearner(BaseLearner):

    def __init__(self, reward_function: RewardFunction):
        super().__init__(reward_function)
        self.context_idxs = None

        self.mu = nn.Sequential(nn.Linear(1, 256), nn.LeakyReLU(), nn.Linear(256, 2))
        self.log_sigma = nn.Sequential(
            nn.Linear(1, 256), nn.LeakyReLU(), nn.Linear(256, 2)
        )

    def reset(self, init_sigma: float = 1.0):
        init_log_sigma = np.log(init_sigma)
        self.context_idxs = None
        self.samples = None
        self.rewards = None
        self.weights = None
        self.mu = nn.Sequential(
            nn.Linear(1, 256), nn.LeakyReLU(), nn.Linear(256, 2)
        )
        self.log_sigma = nn.Sequential(
            nn.Linear(1, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 2),
            ScaleShift(2, 1e-5, init_log_sigma),
        )

    def sample(self, n_samples: int = 10):
        self.context_idxs = np.random.randint(0, 4, n_samples)

        noise = np.random.normal(0, 1, (n_samples, 2))

        with torch.no_grad():
            policy_in = torch.tensor(self.context_idxs, dtype=torch.float).view(-1, 1)
            policy_mu = self.mu.forward(policy_in).detach().cpu().numpy()
            policy_sigma = (
                torch.exp(self.log_sigma.forward(policy_in)).detach().cpu().numpy()
            )
        self.samples = policy_mu + policy_sigma * noise

        self.rewards = None
        self.weights = None

    def evaluate(self):
        self.rewards = self.reward_function.evaluate(
            self.context_idxs, self.samples[:, 0], self.samples[:, 1]
        )
        self.weights = None

    def weigh(self, elite_proportion: float = 0.5):
        # Cross-entropy weighting
        reward_argsort = np.argsort(self.rewards)
        self.weights = np.zeros_like(self.rewards)
        self.weights[
            reward_argsort[-int(self.samples.shape[0] * elite_proportion) :]
        ] = 1

    def learn(self, n_iter: int = 1000, lr: float = 0.01):
        """
        NLL: use negative log-likelihood to optimize mu and sigma
        """
        optimizer = torch.optim.Adam(
            params=list(self.mu.parameters()) + list(self.log_sigma.parameters()), lr=lr
        )
        weights = torch.as_tensor(self.weights, dtype=torch.float)
        for _ in range(n_iter):
            optimizer.zero_grad()
            mu = self.mu.forward(
                torch.tensor(self.context_idxs, dtype=torch.float).view(-1, 1)
            )
            log_sigma = self.log_sigma.forward(
                torch.tensor(self.context_idxs, dtype=torch.float).view(-1, 1)
            )
            dist = torch.distributions.Normal(mu, torch.exp(log_sigma))
            log_probs = dist.log_prob(
                torch.tensor(self.samples, dtype=torch.float)
            ).mean(dim=1)
            loss = -torch.mean(weights * log_probs)
            loss.backward()

            optimizer.step()
