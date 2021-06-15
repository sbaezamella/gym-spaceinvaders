# gym-spaceinvaders

[Space Invaders](https://en.wikipedia.org/wiki/Space_Invaders) is a Japanese shoot'em up arcade game.

# 1 Installation

From repository

```bash
git clone https://github.com/sbaezamella/gym-spaceinvaders.git
cd gym-spaceinvaders
pip install -e .
```

# 2 Game Environment

## 2.1 Observation

Type: Box(72)

These correspond to a (x,y) coordinate of all 36 enemy spaceship relative to the agent.

## 2.2 Actions

The game provides 6 actions to interact with the environment. Move actions into the directions left or right, three different shootings types and a no operation action.

Type: Discrete(6)

| Num | Action       |
| --- | ------------ |
| 0   | No operation |
| 1   | Fire         |
| 2   | Move left    |
| 3   | Move right   |
| 4   | Left fire    |
| 5   | Right fire   |

## 2.3 Reward

Reward correspond to the original's game points for destroying an enemy space ship.
