# Crowd2DMFC.py
#
# Two-dimensional crowd-aversion mean field control
# using finite moments as global features.

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Device & global settings
# ============================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

# ============================================================
# Environment: 2D crowd-aversion MFC
# ============================================================

class Crowd2DEnv:
    """
    dX_t = alpha_t dt + sigma dW_t,  X_t in R^2
    """
    def __init__(
        self,
        T=2.0,
        N=100,
        batch_size=4096,
        sigma=0.4,
        lambda_c=1.0,
        gamma=2.0,
        lambda_T=1,
        lambda_T_final=1000,
        x_target=(1.5, 1.5),
        init_std=0.5,
        kde_bandwidth=0.3,
    ):
        self.T = T
        self.N = N
        self.dt = T / N
        self.batch_size = batch_size
        self.sigma = sigma
        self.lambda_c = lambda_c
        self.gamma = gamma
        self.lambda_T = lambda_T
        self.lambda_T_final = lambda_T_final
        self.kde_bandwidth = kde_bandwidth
        self.init_std = init_std

        self.x_target = torch.tensor(x_target, dtype=torch.float32, device=device)

        # Initial distribution: 2D Gaussian around (0,0)
        x0 = init_std * torch.randn(batch_size, 2, device=device)
        self.x0 = x0

   
    def estimate_density(self, x):
        """
        x: (B, 2) positions at current time.
        returns rho_hat: (B,) estimated density at each particle.

        Uses Gaussian kernel:
            K_h(z) = (2π h^2)^(-1) exp( -|z|^2 / (2 h^2) )
            rho_hat(x_i) = (1/N) Σ_j K_h(x_i - x_j).
        Complexity: O(B^2). For large B, use mini-batches / subsampling.
        """
        B = x.shape[0]
        h = self.kde_bandwidth

        # (B,1,2) - (1,B,2) => (B,B,2)
        diff = x.unsqueeze(1) - x.unsqueeze(0)
        sqdist = (diff ** 2).sum(dim=-1)        # (B,B)
        norm_const = 2.0 * np.pi * (h ** 2)
        K = torch.exp(-sqdist / (2.0 * h ** 2)) / norm_const  # (B,B)

        rho_hat = K.mean(dim=1)  # (B,)
        return rho_hat

    def compute_moments_2d(self, x):
        """
        x: (B, 2) positions
        returns: (6,) moments vector
                 [E[X1], E[X2],
                  E[X1^2], E[X2^2],
                  E[X1 X2],
                  E[|X|^4]]
        """
        # first moments
        m1 = x.mean(dim=0)                 # (2,)  [E[X1], E[X2]]

        # second moments
        m2 = (x ** 2).mean(dim=0)          # (2,)  [E[X1^2], E[X2^2]]
        m12 = (x[:, 0] * x[:, 1]).mean()   # scalar E[X1 X2]

        # higher radial moment
        r2 = (x.pow(2).sum(dim=1))         # |x|^2
        m4 = (r2 ** 2).mean()              # E[|x|^4]

        # pack as one vector (length M = 6)
        moments = torch.cat([
            m1,                # 2
            m2,                # 2
            m12.view(1),       # 1
            m4.view(1),        # 1
        ], dim=0)              # total length 6
        return moments

    # ---------- simulate trajectories and cost ----------
    def simulate(self, policy_net):
        """
        Simulate batch of trajectories under policy_net
        and return (final_positions, cost J(theta)).

        policy_net: nn.Module, input (x, global_moments) -> alpha in R^2.
        """
        x = self.x0.clone()  # (B,2)
        J = torch.zeros(1, device=device)

        for n in range(self.N):
            # t = (n + 0.5) * self.dt

            # compute global moments of current empirical measure
            moments = self.compute_moments_2d(x)        # (6,)
            batch_size = x.size(0)
            tmp_expanded = moments.expand(batch_size, -1)  # (B, 6)

            # input = [local state x, global moments μ_t]
            inp = torch.cat([x, tmp_expanded], dim=1)   # (B, 2 + 6) = (B, 8)
            alpha = policy_net(inp)                     # (B, 2)

            # Brownian increment
            dW = torch.sqrt(torch.tensor(self.dt, device=device)) * \
                 torch.randn(self.batch_size, 2, device=device)

            # crowd density at each position
            rho_hat = self.estimate_density(x)      # (B,)

            # running cost per particle
            control_cost = 0.5 * (alpha ** 2).sum(dim=1)         # (B,)
            #crowd_cost = self.lambda_c * (rho_hat ** self.gamma)
            #target_cost = self.lambda_T * ((x - self.x_target) ** 2).sum(dim=1)

            #running_cost = control_cost + crowd_cost + target_cost
            running_cost = control_cost 
            J += running_cost.mean() * self.dt

            # update state
            x = x + alpha * self.dt + self.sigma * dW

        # terminal cost
        terminal_cost = ((x - self.x_target) ** 2).sum(dim=1)
        J += self.lambda_T_final * terminal_cost.mean()

        return x, J
    
    def simulate_trajectory(self, policy_net, x0=None):
        """
        Simulate full trajectories under the given policy.

        Args:
            policy_net: trained PolicyNet
            x0: optional tensor of shape (B, 2) with initial positions.
                If None, use env.x0 (the large training batch).

        Returns:
            X_traj: (N+1, B, 2) tensor with positions at each time step.
                    X_traj[k] = positions at t_k = k * dt.
        """
        if x0 is None:
            x = self.x0.clone()        # (B_train, 2)
        else:
            x = x0.to(device).clone()  # (B, 2)

        B = x.shape[0]
        X_traj = torch.zeros(self.N + 1, B, 2, device=device)
        X_traj[0] = x

        for n in range(self.N):
            # compute global moments of current empirical measure
            moments = self.compute_moments_2d(x)           # (6,)
            tmp_expanded = moments.expand(B, -1)           # (B, 6)

            inp = torch.cat([x, tmp_expanded], dim=1)      # (B, 8)
            alpha = policy_net(inp)                        # (B, 2)

            dW = torch.sqrt(torch.tensor(self.dt, device=device)) * \
                 torch.randn(B, 2, device=device)

            x = x + alpha * self.dt + self.sigma * dW
            X_traj[n + 1] = x

        return X_traj


# ============================================================
# Policy network: alpha_theta(x, moments)
# ============================================================

class PolicyNet(nn.Module):
    """
    Simple MLP: (x in R^2, global_moments in R^6) -> alpha in R^2.
    """
    def __init__(self, dim_in=8, dim_h=64, dim_out=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_h),
            nn.Tanh(),
            nn.Linear(dim_h, dim_h),
            nn.Tanh(),
            nn.Linear(dim_h, dim_out),
        )

    def forward(self, inp):
        return self.net(inp)


# ============================================================
# Training loop
# ============================================================

def train_crowd_mfc(
    num_iter=3000,
    log_every=200,
    visualize_every=1000,
):
    # hyperparameters
    env = Crowd2DEnv(
        T=2.0,
        N=80,
        batch_size=2048,
        sigma=0.3,
        lambda_c=1.0,
        gamma=2.0,
        lambda_T=0.2,
        lambda_T_final=1000,
        x_target=(1.5, 1.5),
        init_std=0.6,
        kde_bandwidth=0.3,
    )

    policy = PolicyNet(dim_in=8, dim_h=64, dim_out=2).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    loss_history = []

    for it in range(num_iter):
        optimizer.zero_grad()
        _, J = env.simulate(policy)
        J.backward()
        optimizer.step()

        loss_history.append(float(J.item()))

        if it % log_every == 0:
            print(f"Iter {it:5d} | J(theta) ≈ {J.item():.4f}")

        if visualize_every is not None and it > 0 and it % visualize_every == 0:
            with torch.no_grad():
                x_final, J_eval = env.simulate(policy)
            print(f"  [Eval]   J(theta) ≈ {J_eval.item():.4f}")
            plot_particles(x_final, title=f"Iter {it}")

    return env, policy, loss_history


# ============================================================
# Simple visualization
# ============================================================

def plot_particles(x, title="Final positions", x_target=(1.5, 1.5)):
    x = x.detach().cpu().numpy()
    x_tar = np.array(x_target)

    plt.figure(figsize=(4, 4))
    plt.scatter(x[:, 0], x[:, 1], s=3, alpha=0.4, label="Particles")
    plt.scatter([x_tar[0]], [x_tar[1]], c="red", s=40, label="Target")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def plot_loss(loss_history):
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("J(theta)")
    plt.title("Training loss (approximate cost)")
    plt.tight_layout()
    plt.show()

def plot_paths(X_traj, x_target=(1.5, 1.5), max_paths=20, title="Sample trajectories"):
    """
    X_traj: (N+1, B, 2) tensor
    Plots up to `max_paths` particle paths in R^2.
    """
    X_traj = X_traj.detach().cpu()
    N_plus_1, B, _ = X_traj.shape
    n_paths = min(max_paths, B)

    plt.figure(figsize=(5, 5))
    for i in range(n_paths):
        xi = X_traj[:, i, :]           # (N+1, 2)
        plt.plot(xi[:, 0], xi[:, 1], alpha=0.4)

    # mark starts and target
    x0 = X_traj[0]
    plt.scatter(x0[:n_paths, 0], x0[:n_paths, 1], marker="o", s=15,
                label="Initial", alpha=0.8)
    x_tar = np.array(x_target)
    plt.scatter([x_tar[0]], [x_tar[1]], marker="*", s=80, label="Target")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    env, policy, loss_history = train_crowd_mfc(
        num_iter=1200,
        log_every=200,
        #visualize_every=100,
    )

    # Final evaluation & plots
    with torch.no_grad():
        x_final, J_eval = env.simulate(policy)
    print(f"Final evaluated cost J(theta) ≈ {J_eval.item():.4f}")

    plot_loss(loss_history)
    plot_particles(x_final, title="Final positions after training")

    torch.save(
        {
            "model_state": policy.state_dict(),
            "env_params": {
                "T": env.T,
                "N": env.N,
                "sigma": env.sigma,
                "lambda_c": env.lambda_c,
                "gamma": env.gamma,
                "lambda_T": env.lambda_T,
                "lambda_T_final": env.lambda_T_final,
                "x_target": env.x_target.detach().cpu().numpy(),
                "init_std": env.init_std,
                "kde_bandwidth": env.kde_bandwidth,
            },
        },
        "crowd2d_policy.pt",
    )
    print("Saved trained policy to crowd2d_policy.pt")
