# chainer-ppo
Reproduction codes of Proximal Policy Optimization (PPO) with chainer

# About

This  repo is a PPO reproduction codes writen with chainer. [See this original paper](https://arxiv.org/abs/1707.06347) for details

# Training the network

Choose the params and run below command.
The default parameters are set for running in atari environment. 

Example:

```sh
python3 main.py --env-type='atari' 
```
For the detail of the parameters check the code or type 

```sh
python3 main.py --help
```

# Results
## Atari

### Breakout
#### Small model (2 conv layers model)

```sh
$ python3 main.py --env-type='atari' --test-run --model-params=trained_results/atari/breakout/small/final_model --atari-model-size='small'
```

|result|score|
|:---:|:---:|
| ![breakout_small_result](./trained_results/atari/breakout/small/breakout_small_result.gif) |![breakout_small_graph](./trained_results/atari/breakout/small/result.png)|


#### Large model (3 conv layers model)

You can check the performance with below command

```sh
python3 main.py --env-type='atari' --test-run --model-params=trained_results/atari/breakout/large/final_model --atari-model-size='large'
```

|result|score|
|:---:|:---:|
| ![breakout_large_result](./trained_results/atari/breakout/large/breakout_large_result.gif) |![breakout_large_graph](./trained_results/atari/breakout/large/result.png)|

## Mujoco
Sorry in progress...