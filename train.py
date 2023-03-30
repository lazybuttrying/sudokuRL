import torch
import pandas as pd
# from io import StringIO
from environment.env import SudokuEnv
from config import LEFT_TIMES
from model.model import ActorCritic
from model.a2c import run_episode, update_params
import gc
import random
import os
# import wandb


def parsing_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epoch", type=int, default=90000)
    parser.add_argument("--epsilon", type=float, default=0.99)

    args = parser.parse_args()
    args.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    return args


def gpu_preprocess(device):
    for i in range(torch.cuda.device_count()):
        print(f"# DEVICE {i}: {torch.cuda.get_device_name(i)}")
        print("- Memory Usage:")
        print(
            f"  Allocated: {round(torch.cuda.memory_allocated(i)/1024**3,1)} GB")
        print(
            f"  Cached:    {round(torch.cuda.memory_reserved(i)/1024**3,1)} GB\n")

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(device)
    print("Current Device: ", torch.cuda.current_device())
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


if __name__ == "__main__":

    args = parsing_args()
    gpu_preprocess(args.device)

    env = SudokuEnv({"device": args.device})

    model = ActorCritic().to(args.device)
    os.makedirs("asset", exist_ok=True)

    # wandb.init(project="sudoku", entity="koios")
    # wandb.watch(model)

    loss = 9999999
    before_success = 0
    for ep in range(args.epoch):
        log = {
            # "total_loss": [],
            # "actor_loss_x": [],
            # "actor_loss_y": [],
            # "actor_loss_v": [],
            # "critic_loss": [],
            # "length": [],
            # "reward": [],
        }

        state = env.reset()
        # print("before")
        # env.env.printBoard(printing=True)
        # print(ep, "=====================")
        # with torch.autograd.detect_anomaly():
        values, log_probs, rewards, done = run_episode(env, model,
                                                       {"epsilon": args.epsilon})

        log["total_loss"], actor_loss, log["critic_loss"] = update_params(
            model.optimizer, values, log_probs, rewards,
            gamma=model.gamma
        )

        log["length"] = len(rewards)
        log["reward"] = sum(rewards)

        if done:
            print(ep, ep-before_success, (env.x, env.y), log["reward"])
            before_success = ep
            # env.env.printBoard(printing=True)
            print("--------------------------------------------------------")

        del values, rewards, log_probs, actor_loss
        torch.cuda.empty_cache()
        gc.collect()

        # wandb.log(log, step=ep)
        torch.save(model.state_dict(),
                   f"./asset/model_last.pth")
        del log
        gc.collect()
