import numpy as np

# Class for the Ornstein-Uhlenbeck Action Noise
class OUActionNoise:
    def __init__(self, mu=None, sigma=0.2, theta=0.15, dt=1e-2, x0=None) -> None:
        self.mu = np.array(mu) if mu is not None else np.zeros(3)  # Default [0, 0, 0] for [throttle, brake, steer]
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = np.array(x0) if x0 is not None else np.zeros_like(self.mu)

        self.reset()

    def __call__(self):
        # Generate noise for each action dimension
        dW = np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)  # Multi-dimensional noise
        x = self.prev_x + self.theta * (self.mu - self.prev_x) * self.dt + self.sigma * dW

        self.prev_x = x

        return x

    def reset(self):
        self.prev_x = self.x0
