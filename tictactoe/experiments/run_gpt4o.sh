set -ex

# Experiment parameters
OPPONENT_POLICY_NAME="Random"
EXP=$(date +"%Y%m%d")

bash ./tictactoe/scripts/pipeline_gpt4o.sh \
	$OPPONENT_POLICY_NAME \
	$EXP
