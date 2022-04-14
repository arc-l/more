## Interleaving Monte Carlo Tree Search and Self-Supervised Learning for Object Retrieval in Clutter

### Baichuan Huang, Teng Guo, Abdeslam Boularias and Jingjin Yu

Accepted by ICRA 2022

Video illustrating the work:

https://user-images.githubusercontent.com/23622170/134541108-767f2684-b308-43c7-9eb0-2794c633974b.mp4

# Under construction...

## Qucik Start (benchmark on the paper)
Two deep nets should be downloaded from https://drive.google.com/drive/folders/12gmTTyQBxtknXmkyA13aH-7llN0XbYQW?usp=sharing and placed under `more/` as
```
more
└───logs_grasp
│   │   snapshot-post-020000.reinforcement.pth
│   │   file012.txt
└───logs_mcts
│   └───runs
│   │   └───2021-09-02-22-59-train-ratio-1-final
│   │   │   └───lifelong_model-20.pth
```
The model for grasping comes from https://github.com/arc-l/vft.

### PPN (baseline)
Run `bash ppn_main_run.sh`
`Environment(gui=False)` can be changed to `Environment(gui=True)` in `ppn_main.py` for visualization.
### MCTS-50 (baseline)
Change `MCTS_MAX_LEVEL` to 4.
Run `bash mcts_main_run.sh`
Simliar to PPN, `gui` can be toggoled on or off.
We have two environments, the first one is for mimicing the real-world environemnt and the second one is for planning.
Alternatily, you can run `python collect_logs_mcts.py` on 6 processors in parallel (we tested it on PC with 8 processors).
### MORE-50 (proposed)
Change `MCTS_MAX_LEVEL` to 3.
Run `bash more_main_run.sh`
Simliar to MCTS-50, `gui` can be toggoled on or off.

## Collect data to train PPN
We use MCTS to collect training data for PPN.
Change `MCTS_ROLLOUTS` to 300 and `MCTS_EARLY_ROLLOUTS` to 50 in `constants.py`.
Change change `MCTS_MAX_LEVEL` to 4.
Change `cases = glob.glob("test-cases/test/*")` to `cases = glob.glob("test-cases/train/*")` in `collect_logs_mcts.py`.
Change `switches = [0]` to `switches = [0,1,2,3,4]` in `collect_logs_mcts.py`. This step is the data augementation.
Run `python collect_logs_mcts.py`
By default, dataset will be recored under `logs_grasp`. You should move them under `logs_mcts/train`.

## Train PPN
There are two rounds of training.
The first run, `python lifelong_trainer.py --dataset_root 'logs_mcts/train' --ratio 1`.
Then, comment out line 30-35 in `lifelong_trainer.py`, and uncomment line 36-41.
The second run, `python lifelong_trainer.py --dataset_root 'logs_mcts/train' --ratio 1 --pretrained_model 'logs_mcts/runs/PATH_TO_FIRST_RUN/lifelong_model-50.pth'`.