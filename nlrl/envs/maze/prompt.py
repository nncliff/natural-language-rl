SYSTEM_PROMPT = "You are an expert maze solver. You only respond in json."

EVAL_USER_PROMPT_SA_V2 = f"""\
You are playing a simple maze game. An agent is moving in the maze and the objective is to reach the goal in as few steps as possible. 

The possible actions are "move up\n", "move down\n", "move left\n", "move right\n".

You are a highly skilled evaluator in this game. At each step I will provide you with the move history of the agent, where the goal is, your current position, the walls that surround the agent, and the action that the agent is going to take. Your output evaluation should be a json array that includes the following concepts:
- "thoughts": Let's think step by step. Generate your detailed thought process and evaluation of the agent's position after taking the possible action and the distance towards the goal.
- "final_evaluation": Judge how good the agent's current position is after taking the action, compared to all the possible positions in the maze, in terms of reaching the goal.

Let's start a new game. Now, please give your evaluation given action {{chosen_action}} and the current environment state:
```
{{game_content}}
```
""".strip()

EVAL_USER_PROMPT_S_V = f"""\
You are playing a simple maze game. An agent is moving in the maze and the objective is to reach the goal in as few steps as possible. 

The possible actions are "move up\n", "move down\n", "move left\n", "move right\n".

You are a highly skilled evaluator in this game and is expected to function similar to state value function in reinforcement learning. At each step I will provide you with the move history of the agent (from old to new), including where the goal is, your current position, the walls that surround the agent. Your output evaluation should be a json array that includes the following concepts:
- "thoughts": Let's think step by step. Generate your detailed thought process and evaluation of the agent's position and the distance towards the goal.
- "final_evaluation": Concisely judge how good the agent's current position is compared to all the possible positions in the maze, in terms of reaching the goal.

Let's start a new game. Now, please give your evaluation of the current state given the move history of the agent:
```
{{game_content}}
```

""".strip()

EVAL_USER_PROMPT_S_TD_G2 = f"""\
You are playing a simple maze game. An agent is moving in the maze and the objective is to reach the goal in as few steps as possible. 

The possible actions are "move up\n", "move down\n", "move left\n", "move right\n".

You are a highly skilled evaluator in this game, particularly adept at making accurate assessments through look-ahead of the current maze position after taking the given action. At each step I will provide you with the move history of the agent (from old to new), including where the goal is, your current position, the walls that surround the agent, the action that the agent is going to take, *along with several key variations of trajectory pieces after taking this action (and the corresponding natural language evaluations of the trajectory pieces)*. 
Your task is to understand these look-ahead information and summarize, derive non-trivial analysis and understanding the *the agent's position after taking the action*. Your output evaluation should be a json array with the following *two* concepts:
- "thoughts": Let's think step by step. Summarize the look-ahead variations of trajectory pieces and their evaluations.
- "final_evaluation": Now Concisely judge how good the chosen action is, in terms of reaching the goal.

Now, please give your evaluation given action {{chosen_action}}, the *current environment state*:
```
{{game_content}}
```

and the look-ahead information of {{num_variations}} variations after taking action {{chosen_action}}:

```
*Variation 1*, what may happen {{ahead_steps}} look-ahead steps after taking action {{chosen_action}}: {{game_content1}}
```
,
```
Variation 1 evaluation: {{Variation1}}
```
,
```
*Variation 2*, what may happen {{ahead_steps}} look-ahead steps after taking action {{chosen_action}}: {{game_content2}}
```
,
```
Variation 2 evaluation: {{Variation2}}
```
,
```
*Variation 3*, what may happen {{ahead_steps}} look-ahead steps after taking action {{chosen_action}}: {{game_content3}}
```
,
```
Variation 3 evaluation: {{Variation3}}
```
""".strip()

EVAL_USER_PROMPT_S_TD_G2_new = f"""\
You are playing a simple maze game. An agent is moving in the maze and the objective is to reach the goal in as few steps as possible. 

The possible actions are "move up\n", "move down\n", "move left\n", "move right\n".

You are a highly skilled evaluator in this game, particularly adept at making accurate assessments through look-ahead of the current maze position after taking the given action. At each step I will provide you with the move history of the agent (from old to new), including where the goal is, your current position, the walls that surround the agent, the action that the agent is going to take, *along with several key variations of trajectory pieces after taking this action (and the corresponding natural language evaluations of the trajectory pieces)*. 
Your task is to understand these look-ahead information and summarize, derive non-trivial analysis and understanding the *the agent's position after taking the action*. Your output evaluation should be a json array with the following *two* concepts:
- "thoughts": Let's think step by step. Summarize the look-ahead information of the variations after taking action {{chosen_action}}.
- "final_evaluation": Now Concisely judge how good the chosen action is, in terms of reaching the goal.

Now, please give your evaluation given action {{chosen_action}}, the *current environment state*:
```
{{game_content}}
```

and the look-ahead information of different variations after taking action {{chosen_action}}:
""".strip()

POLICY_IMPROVEMENT_PROMPT_TD = f"""\
You are playing a simple maze game. An agent is moving in the maze and the objective is to reach the goal in as few steps as possible. 

Your task is to determine the best action for the next time step given the current state (the move history of the agent (from old to new), including where the goal is, your current position, the walls that surround the agent).

Your possible actions are "move up\n", "move down\n", "move left\n", "move right\n".

The evaluations of the agent after possible actions are given. Each of them consists of two elements:
- "thoughts": Summarization of the look-ahead information of the variations after taking the chosen action.
- "final_evaluation": Judge how good the chosen action is, in terms of reaching the goal.

DO NOT judge the action based on your exterior knowledge, only use the given evaluations to determine the best move.

Here are the evaluations of each possible action:

```
For action "move up", {{evaluations_up}}
```
,
```
For action "move down", {{evaluations_down}}
```
,
```
For action "move left", {{evaluations_left}}
```
,
```
For action "move right", {{evaluations_right}}
```

Return the best action (choose only one from the possible actions) given the evaluations in a json array with a key "action".
""".strip()

POLICY_IMPROVEMENT_PROMPT_SA = f"""\
You are playing a simple maze game. An agent is moving in the maze and the objective is to reach the goal in as few steps as possible. 

Your task is to determine the best action for the next time step given the current state (the move history of the agent (from old to new), including where the goal is, your current position, the walls that surround the agent).

Your possible actions are "move up\n", "move down\n", "move left\n", "move right\n".

The evaluations of the agent after possible actions are given. Each of them consists of two elements:
- "thoughts": Evaluation of the agent's position after taking the possible action and the distance towards the goal.
- "final_evaluation": Judge how good the agent's current position is after taking the action, compared to all the possible positions in the maze, in terms of reaching the goal.

DO NOT judge the action based on your exterior knowledge, only use the given evaluations to determine the best move.

Here are the evaluations of each possible actions:

```
{{evaluations_up}}
```
,
```
{{evaluations_down}}
```
,
```
{{evaluations_left}}
```
,
```
{{evaluations_right}}
```

Return the best action (choose only one from the possible actions) given the evaluations in a json array with a key "action".
""".strip()