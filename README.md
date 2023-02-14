# Retrospect

## Overview of RL Environment
- observation space: `shape(3,9,9,1)`
  - history of 3 step, 9x9 board, 1 cell on each number
- action space: `Discrete(13)`
  - $a<=3$: move (up, down, left, right)
  - $a>=4$: put number (1~9)
- reward: `[-1, 1], -left_times, +left_times`
  - default
    - -0.01 on every movement
    - device count of the correct lines by 27 (total lines to check)
  - truncated
    - When put the number on fixed cell -> reward is -left_times
    - When complete the game -> reward is +left_times




## What I learned

- Reinforcement Learning
  - add recent observation. not use only current state.
  - start from easy level, upgrade to hard level gradually
    - In case of sudoku, higher level, more empty cells.
  - built on model-free
    - I don't tell the rule of sudoku to agent
    - As result, It can fill the empty cell but rarely change the filled numbers (not fixed number). It failed to find the rule.

$ $

- Environment setting
  - git `commit strategy`
  - `faster` when using `ctypes` of python
  - logging training history by `wandb`
  - run on GPU
    - `docker gpu` setting
    - understand pytorch loss `backward calculation process`
    - use pytorch by `cuda`
      - torch.device("cuda:0")
      - torch benchmark = true

## Further Work
- Advantage
- Entropy
- Multi-Action (branching): $a= (x, y, value)$
