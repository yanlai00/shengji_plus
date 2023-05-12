# Playing Shengji with deep reinforcement learning

## Introduction
In recent years, humans have made significant progress in building AIs for perfect and imperfect information games. However, trick-taking poker games are still considered a challenge due to their complexity. Tractor (a.k.a. Tractor) is a 4-player trick-taking card game played with 2 decks of cards that involves competition, collaboration, and state and action spaces that are much larger than the vast majority of existing card games. Currently, there is no existing AI system that can play Tractor. For my Master's thesis, I developed `ShengJi+`, an effective AI system for the Tractor game powered by deep reinforcement learning and Deep Monte Carlo. `ShengJi+` is trained using self-play for ~1.2 million games and achieves 97.7% Level Rate over the random baseline agent. Through case studies, I believe that `ShengJi+` exhibits intelligent and rational playing strategies that resemble human Tractor players.

## Installation
Just clone the repo and make sure you have Pytorch installed in your Python environment. This project doesn't require additional libraries.

## Training
Training the system is as simple as
```shell
python TrainLoop.py --model-folder <SAVE_PATH>
```

There are many options that you can add to this training command. The current best result is produced by the command
```shell
python TrainLoop.py --model-folder <SAVE_PATH> --discount 0.95 --epsilon 0.015 --games 2000 --eval-size 300 --enable-combos --combo-penalty 0.01
```

## Evaluation
You can evaluate the performance of a model by running
```shell
python TrainLoop.py --model-folder <SAVE_PATH> --eval-only --eval-size 3000
```

By default, the random agent is used as opponent. You can also specify the rule-based agent for evaluation by using the `--eval-agent strategic` option. If you would like to play with your model interactively, use the `--eval-agent interactive` option. In that case, you should also add the `--single-process` option so that the games are played in the main process.

To compare two models, you can run
```shell
python TrainLoop.py --eval-only --model-folder my/model1 --compare my/model2 --eval-size 1000
```

You can also play with an agent using the interactive mode:

```shell
python TrainLoop.py --eval-only --model-folder my/model1 --eval-agent interactive --single-process --eval-size 1 --verbose
```

In interactive mode, `ShengJi+` controls the North and South players, and you control East and West. The `--verbose` flag will print out detailed information at each stage of a game.

**Strategic Agent**: In addition to the random agent, we provide a rule-based baseline defined in `agents/StrategicAgent.py`. If you would like to match an agent with this baseline, run 

```shell
python TrainLoop.py --eval-only --model-folder my/model1 --eval-agent strategic --eval-size 1000
```

### Checkpoints
Will be uploaded soon