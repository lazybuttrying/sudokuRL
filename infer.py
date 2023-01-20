import torch
from model.model import ActorCritic
from environment.env import SudokuEnv

import argparse


def parsing_args():
    parser = argparse.ArgumentParser(description="Sudoku A2C Inference")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_path", type=str,
                        default="./asset/model_201_-14.611111111111097.pth")
    parser.add_argument("--trial", type=int, default=200)

    args = parser.parse_args()
    args.device = torch.device(
        "cuda:0" if args.device == "cuda:0" and torch.cuda.is_available() else "cpu")
    return args


if __name__ == "__main__":
    args = parsing_args()

    model = ActorCritic()
    model.load_state_dict(
        torch.load(args.model_path,
                   map_location=args.device))
    model.to(args.device)
    print(
        model.eval()
    )

    env = SudokuEnv({"device": args.device})
    state = env.reset()

    print("=== Game Start ===")
    env.env.printBoard()

    for i in range(args.trial):
        policy, value = model(state.float())
        action = {
            v: torch.distributions.Categorical(
                logits=policy[v].view(-1)).sample()
            for v in ["x", "y", "v"]
        }
        state, reward, _, _, info = env.step(action)
        # env.render()

        print(reward, info)
        env.env.printBoard()
