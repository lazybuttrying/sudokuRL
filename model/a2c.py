import torch
import random


def update_params(optim, values, log_probs, rewards, clc=3, gamma=0.95):
    rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1)
    log_probs = torch.stack(log_probs).flip(dims=(0,)).view(-1)
    values = torch.stack(values).flip(dims=(0,)).view(-1)

    Returns = []
    retrun_ = torch.Tensor([0])
    for r in rewards:
        retrun_ = r + gamma * retrun_
        Returns.append(retrun_)

    Returns = torch.stack(Returns).view(-1)
    actor_loss = torch.sum(-log_probs * (Returns - values.detach()))
    critic_loss = torch.sum(torch.pow(Returns - values, 2))
    loss = actor_loss + \
        critic_loss * clc

    loss.backward()
    optim.step()

    return loss, actor_loss, critic_loss


def run_episode(env, model, params, type="weak"):

    state = env.reset(type=type)

    values = []
    rewards = []
    log_probs = []

    done = False
    new_value = None
    model.optimizer.zero_grad()
    while not done:
        action = torch.distributions.Categorical(
            logits=policy.view(-1)).sample()

        state, reward, truncated, done, info = env.step(action)

        values.append(value)
        rewards.append(reward)
        log_probs.append(policy.view(-1)[action])

        if truncated:
            break

    return values, log_probs, rewards, done
