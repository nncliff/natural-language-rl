# Maze: Language GPI by prompting

Experiments exploring leverage language GPI to improve LLM capability with pure prompting. We consider the fully-observable setting, where the agent's observation (described by text) includes the agent's current position in the maze, the agent's action history, the walls' position around the agent (if any), and the goal position. The action space is discrete, including moving up / down / right / left.

The code is adopted from [LMRL Gym](https://github.com/abdulhaim/LMRL-Gym). Please follow the original instructions for environment installation.

```bash
export PYTHONPATH="${PYTHONPATH}:.."
export OPENAI_API_KEY='your-api-key-here'
```

## Usage
```bash
python gpt4/nlrl_maze.py --n_interactions 30 --td --num_variations 8 --ahead_steps 3 --maze_name 'double_t_maze'
```
```bash
python gpt4/nlrl_maze.py --n_interactions 30 --td --num_variations 8 --ahead_steps 3 --maze_name 'maze2d_medium'
```
