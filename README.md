# Natural Language Reinforcement Learning
The official implementation of [Natural Language Reinforcement Learning](https://arxiv.org/abs/2411.14251).

Reinforcement Learning (RL) mathematically formulates decision-making with Markov Decision Process (MDP). With MDPs, researchers have achieved remarkable breakthroughs across various domains, including games, robotics, and language models. This paper seeks a new possibility, Natural Language Reinforcement Learning (NLRL), by extending traditional MDP to natural language-based representation space. Specifically, NLRL innovatively redefines RL principles, including task objectives, policy, value function, Bellman equation, and policy iteration, into their language counterparts. With recent advancements in large language models (LLMs), NLRL can be practically implemented to achieve RL-like policy and value improvement by either pure prompting or gradient-based training. Experiments over Maze, Breakthrough, and Tic-Tac-Toe games demonstrate the effectiveness, efficiency, and interpretability of the NLRL framework among diverse use cases.
## Installation
```
conda create -n nlrl python==3.10
conda activate nlrl

git clone https://github.com/waterhorse1/Natural-language-RL
cd Natural-language-RL
pip install -r requirements.txt

# Install gym-tictactoe (modified from https://github.com/haje01/gym-tictactoe)
pip3 install -e tictactoe/gym-tictactoe/.

# Install Forked OpenSpiel
git clone https://github.com/waterhorse1/open_spiel
cd open_spiel
pip3 install -e .

# We use a hacky implementation that reuse Huggingface checkpoint loading
# to reload model and optimizer at each training iteration
# which needs to slightly modify the original code of HuggingFace Trainer
# Comment line 1912 - 1914 in anaconda3/envs/nlrl/lib/python3.10/site-packages/transformers/trainer.py
# Specifically comment out these 3 lines:
state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
if state.train_batch_size is not None:
    self._train_batch_size = state.train_batch_size

# Add current dir to PYTHONPATH
export PYTHONPATH=./
```
## Open-Source Datasets and Models
### Datasets
We open-source the breakthrough's language TD training set and test set on Huggingface: https://huggingface.co/datasets/Waterhorse/Breakthrough_dataset
### Open Source Model
We open-source our trained policy and value network on Huggingface：\
* Breakthrough:
  - Language Value: https://huggingface.co/Waterhorse/Llama-3.1-8B-Instruct-NLRL-Breakthrough-Value
* TicTacToe:
	- Language Policy: https://huggingface.co/Benjamin-eecs/Llama-3.1-8B-Instruct-NLRL-TicTacToe-Policy
	- Language Value: https://huggingface.co/Benjamin-eecs/Llama-3.1-8B-Instruct-NLRL-TicTacToe-Value
## Usage
The `nlrl` directory contains shared libraries for our three experiments: `maze`, `breakthrough`, and `tictactoe`. Please refer to the README in each respective folder for specific usage instructions.
## Citation
```bibtex
@misc{nlrl,
      title={Natural Language Reinforcement Learning},
      author={Xidong Feng and Ziyu Wan and Haotian Fu and Bo Liu and Mengyue Yang and Girish A. Koushik and Zhiyuan Hu and Ying Wen and Jun Wang},
      year={2024},
      eprint={2411.14251},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2411.14251},
}
```
