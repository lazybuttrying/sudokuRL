import torch
# import pandas as pd
# from io import StringIO
from environment.env import SudokuEnv
from model.model import ActorCritic
from model.a2c import run_episode, update_params
import gc
import random
import wandb

import argparse


def parsing_args():
    parser = argparse.ArgumentParser(description="Sudoku A2C Train")
    parser.add_argument("--epoch", type=int, default=99999)
    parser.add_argument("--epsilon", type=float, default=0.99)
    parser.add_argument("--device", type=str, default="cuda:0")
    # parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--gamma", type=float, default=0.95)

    args = parser.parse_args()
    args.device = torch.device(
        "cuda:0" if args.device == "cuda:0" and torch.cuda.is_available() else "cpu")
    return args


def gpu_preprocess():
    for i in range(torch.cuda.device_count()):
        print(f"# DEVICE {i}: {torch.cuda.get_device_name(i)}")
        print("- Memory Usage:")
        print(
            f"  Allocated: {round(torch.cuda.memory_allocated(i)/1024**3,1)} GB")
        print(
            f"  Cached:    {round(torch.cuda.memory_reserved(i)/1024**3,1)} GB\n")

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(DEVICE)
    print("Current Device: ", torch.cuda.current_device())
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


if __name__ == "__main__":

    args = parsing_args()
    gpu_preprocess()

    env = SudokuEnv({"device": DEVICE})

    model = ActorCritic().to(DEVICE)

    wandb.init(project="sudoku", entity="koios")
    wandb.watch(model)

    state = env.reset()
    best = -500
    for ep in range(args.epoch):
        if best > 100:
            print("Well Done!")
            break

        log = {
            "total_loss": 0,
            "actor_loss_x": 0,
            "actor_loss_y": 0,
            "actor_loss_v": 0,
            "critic_loss": 0,
            "length": 0,
            "reward": 0,
            "best": best
        }

        values, log_probs, rewards = run_episode(env, model,
                                                 {"epsilon": args.epsilon})

        log["total_loss"], actor_loss, log["critic_loss"] = update_params(
            model.optimizer, values, log_probs, rewards,
            gamma=model.gamma
        )
        log["actor_loss_x"] = actor_loss["x"]
        log["actor_loss_y"] = actor_loss["y"]
        log["actor_loss_v"] = actor_loss["v"]

        log["length"] = len(rewards)
        log["reward"] = sum(rewards)

        # log["last_board"] = pd.read_csv(StringIO(env.env.printBoard(printing=False)), sep=",")

        del values, rewards, log_probs, actor_loss, total_loss, critic_loss
        torch.cuda.empty_cache()
        gc.collect()

        if log["reward"] > log["best"]:
            torch.save(model.state_dict(),
                       f"./asset/model_{ep}_{log['best']}.pth")
            log["best"] = log["reward"]

        wandb.log(log, step=ep)
        best = log["best"]
        del log
        gc.collect()
