# simulate_particles.py
#
# Use a previously trained crowd-aversion NN control to simulate particles
# and visualize their dynamics as an animation.

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Import your environment, policy and device from the training file
from Crowd2DMFC import Crowd2DEnv, PolicyNet, device


# ------------------------------------------------------------
# Animation
# ------------------------------------------------------------

def animate_particles_with_paths(X_traj, x_target=(1.5, 1.5),
                                 interval=150,
                                 save_path=None):
    """
    X_traj : (T_steps, B, 2) tensor from simulate_trajectory
    Plots both current positions (scatter) and full trajectories (lines).
    """
    X = X_traj.detach().cpu().numpy()
    T_steps, B, _ = X.shape
    x_tar = np.array(x_target)

    fig, ax = plt.subplots(figsize=(5, 5))

   
    lines = [
        ax.plot([], [], lw=1.5, alpha=0.7)[0]   # Line2D objects
        for _ in range(B)
    ]
    # Scatter for current positions
    scat = ax.scatter([], [], s=40, alpha=0.9, c="C0", label="Particles")
    ax.scatter([x_tar[0]], [x_tar[1]], marker="*", s=80, c="red", label="Target")

    # Axis limits based on all positions
    x_min = X[..., 0].min()
    x_max = X[..., 0].max()
    y_min = X[..., 1].min()
    y_max = X[..., 1].max()
    pad = 0.5
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()

    def init():
        # no trajectory yet
        for line in lines:
            line.set_data([], [])
        scat.set_offsets(np.empty((0, 2)))
        ax.set_title("t = 0.00 T")
        # return all to update
        return lines + [scat]

    def update(frame):
        # positions at current frame
        positions = X[frame]  # (B, 2)

        # update each particle's path: all points from time 0 to current frame
        for i, line in enumerate(lines):
            x_path = X[:frame+1, i, 0]
            y_path = X[:frame+1, i, 1]
            line.set_data(x_path, y_path)

        # update scatter for current positions
        scat.set_offsets(positions)

        tau = frame / (T_steps - 1)
        ax.set_title(f"t = {tau:.2f} T")
        return lines + [scat]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=T_steps,
        init_func=init,
        blit=True,
        interval=interval,
    )

    if save_path is not None:
        anim.save(save_path, fps=8, dpi=150)
        print(f"Animation saved to {save_path}")

    return anim

# ------------------------------------------------------------
# Main: load policy, rebuild env, simulate particles
# ------------------------------------------------------------

if __name__ == "__main__":
    # 1) Load checkpoint saved after training
    checkpoint_path = "crowd2d_policy.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state"]
    env_params = checkpoint["env_params"]

    # 2) Rebuild environment (set batch_size=5 for 5 paths)
    env = Crowd2DEnv(
        T=env_params["T"],
        N=env_params["N"],
        batch_size=5,  # number of particles
        sigma=env_params["sigma"],
        lambda_c=env_params["lambda_c"],
        gamma=env_params["gamma"],
        lambda_T=env_params["lambda_T"],
        lambda_T_final=env_params["lambda_T_final"],
        x_target=tuple(env_params["x_target"]),
        init_std=env_params["init_std"],
        kde_bandwidth=env_params["kde_bandwidth"],
    )

    # 3) Rebuild PolicyNet with the same architecture used in training
    policy = PolicyNet(dim_in=8, dim_h=64, dim_out=2).to(device)
    policy.load_state_dict(state_dict)
    policy.eval()
    print("Loaded trained policy from", checkpoint_path)

    # 4) Simulate trajectories of 5 new particles under the loaded control
    with torch.no_grad():
        # Sample 5 new initial positions from the same initial law
        x0_5 = env.init_std * torch.randn(5, 2, device=device)

        # This method must exist in Crowd2DEnv (which depends on the specific problem)
        X_traj_5 = env.simulate_trajectory(policy, x0=x0_5)

    # 5) Animate
    anim = animate_particles_with_paths(
        X_traj_5,
        x_target=env.x_target.tolist(),
        interval=150,
        save_path=None,         # or "crowd_5particles.mp4"
    )

    # Show the animation window (for scripts run locally)
    plt.show()