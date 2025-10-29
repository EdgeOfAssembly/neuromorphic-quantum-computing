import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np
import math
import time

# Fixed seed
torch.manual_seed(42)
np.random.seed(42)

class CartPoleEnv:
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.state = None
        
    def reset(self):
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        return self.state, {}
    
    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4/3 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        
        self.state = np.array([x, x_dot, theta, theta_dot])
        
        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        reward = 1.0
        truncated = False
        info = {}
        return self.state, reward, done, truncated, info

class QTUNLayer(nn.Module):
    def __init__(self, in_features, out_features, threshold=0.01):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.threshold = threshold

    def qrelu(self, pre_act):
        collapsed = torch.zeros_like(pre_act)
        pos_mask = pre_act > self.threshold
        neg_mask = pre_act < -self.threshold
        super_mask = ~(pos_mask | neg_mask)
        collapsed[pos_mask] = 1.0
        collapsed[neg_mask] = 0.0

        if super_mask.any():
            collapsed[super_mask] = F.softplus(pre_act[super_mask])

        return collapsed

    def forward(self, x):
        pre_act = torch.matmul(x, self.weights.t())
        act = self.qrelu(pre_act)
        return act

class QTUNActorCritic(nn.Module):
    def __init__(self, state_dim=4, hidden_dims=[256, 256], action_dim=2, threshold=0.01):
        super().__init__()
        layers = []
        prev_dim = state_dim
        for h in hidden_dims:
            layers.append(QTUNLayer(prev_dim, h, threshold))
            prev_dim = h
        self.shared_layers = nn.ModuleList(layers)
        self.actor_head = nn.Linear(prev_dim, action_dim)
        self.critic_head = nn.Linear(prev_dim, 1)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.normal_(self.actor_head.weight, mean=0, std=0.1)
        nn.init.zeros_(self.critic_head.bias)
        nn.init.normal_(self.critic_head.weight, mean=0, std=0.1)
        
    def forward(self, x, temperature=1.0):
        for layer in self.shared_layers:
            x = layer(x)
        logits = self.actor_head(x) / temperature
        probs = F.softmax(logits, dim=-1)
        value = self.critic_head(x).squeeze(-1)
        return probs, value
    
    def get_entropy(self, probs):
        return -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

def compute_gae(rewards, values, next_value, gamma=0.99, lam=0.95):
    """Full GAE (no truncation)."""
    advantages = []
    gae = 0
    for r, v in zip(reversed(rewards), reversed(values)):
        delta = r + gamma * next_value - v
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        next_value = v
    return torch.tensor(advantages)

def compute_ece(probs_list, rewards_list, bins=10):
    if not probs_list:
        return 0.0
    probs = torch.cat(probs_list, dim=0)
    rewards = torch.tensor(rewards_list, dtype=torch.float32)
    conf = 1 - (-(probs * torch.log(probs + 1e-8)).sum(-1))
    norm_rewards = (rewards - rewards.min()) / (rewards.max() - rewards.min() + 1e-8)
    bin_bounds = torch.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        in_bin = (conf >= bin_bounds[i]) & (conf < bin_bounds[i + 1])
        if in_bin.any():
            acc_in_bin = norm_rewards[in_bin].mean()
            conf_in_bin = conf[in_bin].mean()
            ece += (bin_bounds[i + 1] - bin_bounds[i]) * abs(acc_in_bin - conf_in_bin)
    return ece.item()

def train_qtun_a2c_cartpole(episodes=1500, lambda_entropy=0.01, lr=3e-4, temperature=1.0):
    env = CartPoleEnv()
    state_dim = 4
    action_dim = 2
    
    model = QTUNActorCritic(state_dim, hidden_dims=[256, 256])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    episode_rewards = []
    episode_entropies = []
    episode_ece = []
    start_time = time.time()
    solved = False
    stall_count = 0
    stall_threshold = 200  # Stop if avg <10 for this many eps
    
    for ep in range(episodes):
        # LR decay
        if ep % 500 == 0 and ep > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
        
        state, _ = env.reset()
        log_probs = []
        values = []
        rewards = []
        entropies = []
        probs_list = []
        done = False
        
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            probs, value = model(state_t, temperature)
            dist = Categorical(probs=probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            next_state, reward, done, _, _ = env.step(action.item())
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            entropies.append(model.get_entropy(probs).item())
            probs_list.append(probs)
            
            state = next_state
        
        next_value = 0.0 if done else model(torch.FloatTensor(next_state).unsqueeze(0), temperature)[1].item()
        advantages = compute_gae(rewards, [v.item() for v in values], next_value)
        adv_mean = advantages.mean()
        adv_std = advantages.std().clamp(min=1e-4)
        advantages = (advantages - adv_mean) / adv_std
        returns = advantages + torch.tensor([v.item() for v in values])
        
        policy_loss = 0
        value_loss = 0
        for log_prob, advantage, ret, value in zip(log_probs, advantages, returns, values):
            policy_loss += -log_prob * advantage
            value_loss += F.mse_loss(value, ret.unsqueeze(0))
        policy_loss = policy_loss.mean()
        value_loss = value_loss.mean()
        
        # Dynamic entropy: Boost if low
        avg_ent = np.mean(entropies)
        dynamic_lambda = lambda_entropy
        if avg_ent < 0.4:
            dynamic_lambda *= 2  # Temp boost exploration
        ent_loss = -dynamic_lambda * torch.tensor(entropies).mean()
        loss = policy_loss + 0.5 * value_loss + ent_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        ep_reward = sum(rewards)
        episode_rewards.append(ep_reward)
        episode_entropies.append(avg_ent)
        
        # Stall check
        recent_avg = np.mean(episode_rewards[-100:])
        if recent_avg < 10:
            stall_count += 1
        else:
            stall_count = 0
        if stall_count >= stall_threshold:
            print(f"Stalled at Ep {ep} (avg {recent_avg:.1f})—stopping early.")
            break
        
        # Prints: Separate <10 and %100 to avoid dup
        if ep < 10:
            ece = compute_ece(probs_list, rewards)
            episode_ece.append(ece)
            elapsed = time.time() - start_time
            print(f"Ep {ep}: Reward {ep_reward:.1f}, Avg Ent {avg_ent:.3f}, ECE {ece:.3f}, Time so far {elapsed:.1f}s")
        if ep % 100 == 0 and ep > 0:
            ece = compute_ece(probs_list, rewards)
            episode_ece.append(ece)
            elapsed = time.time() - start_time
            print(f"Ep {ep}: Reward {ep_reward:.1f}, Avg Ent {avg_ent:.3f}, ECE {ece:.3f}, Time so far {elapsed:.1f}s")
            if recent_avg >= 195 and not solved:
                print(f"SOLVED at Ep {ep}! Avg {recent_avg:.1f}")
                solved = True
    
    total_time = time.time() - start_time
    final_avg_reward = np.mean(episode_rewards[-100:])
    final_avg_ece = np.mean(episode_ece[-10:]) if len(episode_ece) > 10 else episode_ece[-1] if episode_ece else 0
    print(f"Final Avg Reward (last 100): {final_avg_reward:.1f}")
    print(f"Final Avg ECE: {final_avg_ece:.3f}")
    print(f"Total time for {ep} episodes: {total_time:.1f}s")
    if not solved:
        print("Not yet solved—try more episodes or tweaks!")
    
    return episode_rewards, episode_ece

# Run the test
if __name__ == "__main__":
    rewards, eces = train_qtun_a2c_cartpole()