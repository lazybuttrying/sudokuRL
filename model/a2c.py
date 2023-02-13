import torch
import random


def update_params(optim, values, log_probs, rewards, clc=3, gamma=0.95):
    rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1)
    # log_probs = {v: torch.stack(log_probs[v]).flip(dims=(0,)).view(-1)
    #              for v in ["x", "y", "v"]}
    log_probs = torch.stack(log_probs).flip(dims=(0,)).view(-1)
    values = torch.stack(values).flip(dims=(0,)).view(-1)

    Returns = []
    retrun_ = torch.Tensor([0])
    for r in rewards:
        retrun_ = r + gamma * retrun_
        Returns.append(retrun_)

    Returns = torch.stack(Returns).view(-1)
    actor_loss = torch.sum(-log_probs * (Returns - values.detach()))
    # actor_loss = {
    #     v: torch.sum(-log_prob * (Returns - values.detach()))
    #     for v, log_prob in log_probs.items()
    # }
    critic_loss = torch.sum(torch.pow(Returns - values, 2))
    # loss = actor_loss["x"] + actor_loss["y"] + actor_loss["v"] + \
    loss = actor_loss + \
        critic_loss * clc
    loss.backward()
    optim.step()

    return loss, actor_loss, critic_loss


def run_episode(env, model, params, type="weak"):

    state = env.reset(type=type)

    # print(state.shape)

    values = []
    rewards = []
    log_probs = []
    # log_probs = {
    #     "x": [],
    #     "y": [],
    #     "v": []
    # }

    done = False
    new_value = None
    model.optimizer.zero_grad()
    while not done:

        # policy, value = model(state.float().cuda())
        policy, value = model(state.float())

        if random.random() > params["epsilon"]:
            action = env.action_space.sample()
        else:
            action = torch.distributions.Categorical(
                logits=policy.view(-1)).sample()
            # action = {
            #     v: torch.distributions.Categorical(
            #         logits=policy[v].view(-1)).sample()
            #     for v in ["x", "y", "v"]
            # }
        state, reward, truncated, done, info = env.step(action)

        values.append(value)
        rewards.append(reward)
        log_probs.append(policy.view(-1)[action])
        # for v in ["x", "y", "v"]:
        #     log_probs[v].append(policy[v].view(-1)[action[v]])

        if truncated:
            done = True

    return values, log_probs, rewards
