# Interleaving Monte Carlo Tree Search and Self-Supervised Learning for Object Retrieval in Clutter

**Abstract.** In this study, working with the task of object retrieval in clutter, we have developed a robot learning framework in which Monte Carlo Tree Search (MCTS) is first applied to enable a Deep Neural Network (DNN) to learn the intricate interactions between a robot arm and a complex scene containing many objects, allowing the DNN to partially clone the behavior of MCTS. In turn, the trained DNN is integrated into MCTS to help guide its search effort. We call this approach learning-guided Monte Carlo tree search for Object REtrieval (MORE), which delivers significant computational efficiency gains and added solution optimality. MORE is a self-supervised robotics framework/pipeline capable of working in the real world that successfully embodies the System 2 to System 1 learning philosophy proposed by Kahneman, where learned knowledge, used properly, can help greatly speed up a time-consuming decision process over time.

[YouTube](https://youtu.be/hLYKq8JMDqE)&nbsp;&nbsp;•&nbsp;&nbsp;[PDF](https://arxiv.org/abs/2202.01426)&nbsp;&nbsp;•&nbsp;&nbsp;International Conference on Robotics and Automation (ICRA) 2022

*[Baichuan Huang](https://baichuan05.github.io/), Teng Guo, [Abdeslam Boularias](http://rl.cs.rutgers.edu/abdeslam.html), [Jingjin Yu](http://jingjinyu.com/)*

Video with sound illustrating the work (high-quality video can be access at [YouTube](https://youtu.be/hLYKq8JMDqE)):

https://user-images.githubusercontent.com/20850928/163514798-2bd29c01-a1a8-40ae-8632-b052d2cdfb47.mp4


## Installation (for Ubuntu 18.04)
Recommended: install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

```shell
git clone https://github.com/arc-l/more.git
cd more
conda env create --name more --file=env-more.yml
conda activate more
```

## Quick Start (benchmarking as presented in the paper)
Two deep nets should be downloaded from https://drive.google.com/drive/folders/12gmTTyQBxtknXmkyA13aH-7llN0XbYQW?usp=sharing and placed under `more/` as
```shell
more
└───logs_grasp
│   ├── snapshot-post-020000.reinforcement.pth
└───logs_mcts
│   └───runs
│   │   └───2021-09-02-22-59-train-ratio-1-final
│   │   │   ├── lifelong_model-20.pth
```
The model for grasping comes from https://github.com/arc-l/vft.

### PPN (baseline)
* Run `bash ppn_main_run.sh`
* `Environment(gui=False)` can be changed to `Environment(gui=True)` in `ppn_main.py` for visualization purpose.
* Put all logs in a single folder.
* Run `python evaluate.py --log 'PATH_TO_FOLDER_OF_PPN_RECORDS'` to get benchmark result.

### MCTS-50 (baseline)
* Change `MCTS_MAX_LEVEL` to 4.
* Run `bash mcts_main_run.sh`
* Simliar to PPN, `gui` can be toggled on or off for visualization purpose.
* We have two environments, the first one is for mimicing the real-world environemnt and the second one is for planning.
* Alternatily, you can run `python collect_logs_mcts.py` on 6 processors in parallel (we tested it on PC with 8 processors).
* Put all logs in a single folder.
* Run `python evaluate.py --log 'PATH_TO_FOLDER_OF_MCTS_RECORDS'` to get benchmark result.

### MORE-50 (proposed)
* Change `MCTS_MAX_LEVEL` to 3.
* Run `bash more_main_run.sh`
* Simliar to MCTS-50, `gui` can be toggoled on or off for visualization purpose.
* Put all logs in a single folder.
* Run `python evaluate.py --log 'PATH_TO_FOLDER_OF_MORE_RECORDS'` to get benchmark result.

With GUI on, you should expect to see something like this (video has been shortened):

https://user-images.githubusercontent.com/20850928/163526921-5f5ec4b2-78ed-4f73-9276-4a43acd4dafe.mp4



## Collect data to train PPN
We use MCTS to collect training data for PPN.
* Change `MCTS_ROLLOUTS` to 300 and `MCTS_EARLY_ROLLOUTS` to 50 in `constants.py`.
* Change change `MCTS_MAX_LEVEL` to 4.
* Change `cases = glob.glob("test-cases/test/*")` to `cases = glob.glob("test-cases/train/*")` in `collect_logs_mcts.py`.
* Change `switches = [0]` to `switches = [0,1,2,3,4]` in `collect_logs_mcts.py`. This step is the data augementation.
* Run `python collect_logs_mcts.py`
* By default, dataset will be recored under `logs_grasp`. You should move them under `logs_mcts/train`.


## Train PPN
There are two rounds of training.
* The first run, `python lifelong_trainer.py --dataset_root 'logs_mcts/train' --ratio 1`.
Then, comment out line 30-35 in `lifelong_trainer.py`, and uncomment line 36-41.
* The second run, `python lifelong_trainer.py --dataset_root 'logs_mcts/train' --ratio 1 --pretrained_model 'logs_mcts/runs/PATH_TO_FIRST_RUN/lifelong_model-50.pth'`.


## Citing MORE
If this work helps your research, please cite the [MORE](https://arxiv.org/abs/2202.01426):

```
@inproceedings{huang2022interleaving,
  title        = {Interleaving Monte Carlo Tree Search and Self-Supervised Learning for Object Retrieval in Clutter},
  author       = {Huang, Baichuan and Guo, Teng and Boularias, Abdeslam and Yu, Jingjin},
  booktitle    = {2022 IEEE International Conference on Robotics and Automation (ICRA)},
  year         = {2022},
  organization = {IEEE}
}
```

This work also builds on many other papers. We found the following resources are helpful!
* https://github.com/google-research/ravens
* https://github.com/andyzeng/visual-pushing-grasping
* https://github.com/arc-l/vft
* https://github.com/arc-l/dipn
