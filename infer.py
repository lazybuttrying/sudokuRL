# %%
import torch
from model.model import ActorCritic
from environment.env import SudokuEnv


def parsing_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--iter", type=int, default=90000)
    parser.add_argument("--board", type=str, default="./asset/board.pkl")
    parser.add_argument("--model_file", type=str,
                        default="./asset/model_last.pth")

    args = parser.parse_args()
    args.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    return args


args = parsing_args()

model = ActorCritic()
model.load_state_dict(
    torch.load(args.model_file, map_location=args.device))
model.to(args.device)
print(
    model.eval()
)

# %%
env = SudokuEnv({"device": args.device})
state = env.reset(type="infer")

print("=== Game Start ===")
env.env.printBoard()

for i in range(args.iter):
    policy, value = model(state.float())
    action = {
        v: torch.distributions.Categorical(
            logits=policy[v].view(-1)).sample()
        for v in ["x", "y", "v"]
    }
    # print(action)
    state, reward, truncated, done, info = env.step(action)
    # env.render()

    if truncated:
        print("Truncated")
        break
    if reward > -0.01 and i % 10:
        print(reward, i, info)  # action["x"], action["y"], action["v"])
        env.env.printBoard()
