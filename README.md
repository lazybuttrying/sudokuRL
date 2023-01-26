# Retrospect

- use pytorch by ```cuda```
  - torch.device("cuda:0")
  - torch benchmark = true
- ```docker gpu``` setting
- ```faster``` when using ```ctypes``` of python
- git ```commit strategy```
- understand pytorch loss ```backward calculation process```
- ```wandb```


# Wonder
Initial trajectory is not good for learning, especially sudoku. To get score, agent has to fill the black. But untill filling whole blank, value awalys goes down. 
My point is that how about starting after empty-initial state? 
As case like sudoku, agent doesn't start at blank board, state filled all the blank
