## Examples

This README walks you through all steps to run an external gym environment.
The example will use this repository [gym-spaceinvaders](https://github.com/sbaezamella/gym-spaceinvaders) as this is part of this repository, but never the less this works similar with other external gym environments.

### 1 Install the additional package

You need to clone the repository and install the package as follows:

```bash
git clone https://github.com/sbaezamella/gym-spaceinvaders
cd gym-spaceinvaders
pip install -e .
```

### 2 Import packages in your code

To use an external gym environment you allways need to import the corresponding package along with the regular gym package.

```python
import gym
import gym_spaceinvaders
```

### 3 Load the environment

From now on everything is as you are used to it. You can simply make the environment, render it, perform actions and so on.

```Python
env = gym.make('CustomSpaceInvaders-v0')

action = env.action_space.sample()
observation, reward, done, info = env.step(action)
```

Now that you are all set with the preparations enjoy the external environment.
