import os
import json
from aibread import Bread

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


client = Bread(
    api_key=os.environ.get("BREAD_API_KEY")
)

repos = client.repo.list()

print(repos)

# Create repository
repo = client.repo.set(repo_name="zip5")
# Load tools and messages from JSON file
with open('tools_and_messages.json', 'r') as f:
    data = json.load(f)
    tools_list = data['tools']
    messages_list = data['messages']

# Create teacher prompt
client.prompts.set(
    prompt_name="system_prompt",
    repo_name="zip5",
    messages=[
        {
            "role": "system",
            "content": "You are Yoda. Speak like Yoda, use inverted syntax, few words, and wise, cryptic tone, always calm and reflective."
        }
    ],
)

# Create student prompt
client.prompts.set(
    prompt_name="user_prompt",
    repo_name="zip5",
    messages=[]
)

target = client.targets.set(
    target_name="coding_target9",
    repo_name="zip5",
    template="default",
    overrides={
        "generators": [
            {
                "type": "oneshot_qs",
                "numq": 250,
                "model": "claude-sonnet-4-5-20250929"
            }
        ],
        "model_name": "Qwen/Qwen3-32B",
        "u": "system_prompt",
        "v": "user_prompt",
        "num_traj_per_stimulus": 1
        # "temperature": 1.2
    }
)

import time

# Start stim job
client.targets.stim.run(
    target_name="coding_target9",
    repo_name="zip5"
)

print("running stim")

# Poll until complete
while True:
    status = client.targets.stim.get(
        target_name="coding_target9",
        repo_name="zip5"
    )
    if status.status == "complete":
        print(f"Stim complete! Generated {status.lines} stimuli")
        break
    print(f"Status: {status.status}")
    time.sleep(5)

print("complete")

import time

# Start rollout job
client.targets.rollout.run(
    target_name="coding_target9",
    repo_name="zip5"
)

# Poll until complete
while True:
    status = client.targets.rollout.get(
        target_name="coding_target9",
        repo_name="zip5"
    )
    if status.status == "complete":
        print(f"Rollout complete! Generated {status.lines} trajectories")
        break
    print(f"Status: {status.status}")
    time.sleep(5)


#Compose your dataset and training config
client.bakes.set(
    bake_name="my_bake7",
    repo_name="zip5",
    template="default",
    overrides={
        "datasets": [
            {"target": "coding_target9", "weight": 1.0}
        ],
        "epochs": 1,
        "micro_batch_size": 1
    }
)

import time

# Start bake job
client.bakes.run(bake_name="my_bake7", repo_name="zip5")

# Monitor progress
while True:
    status = client.bakes.get(bake_name="my_bake7", repo_name="zip5")
    if status.status == "complete":
        print("Training complete!")
        break
    time.sleep(30)