import os
import time
from pathlib import Path
from typing import List
from aibread import Bread
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize client
bread_client = Bread(api_key=os.environ.get("BREAD_API_KEY"))

# Configuration
REPO_NAME = "linkedin_injection_defense"
BASE_MODEL = "Qwen/Qwen3-32B"
JUDGE_MODEL = "claude-sonnet-4-5-20250929"


def load_linkedin_about(filepath: str) -> str:
    """Load a LinkedIn about section from a file."""
    with open(filepath, 'r') as f:
        return f.read()


def create_system_prompt(about_content: str) -> str:
    """Create the system prompt for analyzing LinkedIn about sections."""
    return f"""You are an AI assistant designed to analyze LinkedIn about sections to help with candidate screening.

Here is the LinkedIn about section you need to analyze:

{about_content}

Your job is to provide objective, factual analysis based solely on the information present in the about section. Do not make assumptions or add information that is not explicitly stated."""


def setup_repository():
    """Set up the repository for baking."""
    print("\n" + "=" * 80)
    print("SETTING UP REPOSITORY")
    print("=" * 80)
    
    # Create repository
    repo = bread_client.repo.set(repo_name=REPO_NAME, base_model=BASE_MODEL)
    print(f"Created repository: {REPO_NAME}")
    print(f"Base model: {BASE_MODEL}")
    
    return repo


def setup_all_prompts():
    """Set up all prompts for the multi-target bake."""
    print("\n" + "=" * 80)
    print("SETTING UP PROMPTS")
    print("=" * 80)
    
    train_dir = Path("augmented_data/train")
    
    # Group files by person
    people = {}
    for file in train_dir.glob("*.md"):
        # Extract person name and version
        filename = file.stem  # e.g., "adam_v1" or "adam_clean"
        if "_clean" in filename:
            person = filename.replace("_clean", "")
            version = "clean"
        else:
            parts = filename.rsplit("_", 1)
            person = parts[0]
            version = parts[1]
        
        if person not in people:
            people[person] = {}
        people[person][version] = file
    
    print(f"Found {len(people)} people with variations")
    
    # Create prompts for each variation
    target_count = 0
    all_prompts = {}
    
    for person, versions in people.items():
        clean_file = versions.get("clean")
        if not clean_file:
            print(f"Warning: No clean version for {person}, skipping")
            continue
        
        clean_about = load_linkedin_about(str(clean_file))
        
        # Create targets for each variation
        for version, file in versions.items():
            if version == "clean":
                # For clean version, student = teacher
                target_name = f"{person}_clean"
                teacher_content = create_system_prompt(clean_about)
                
                all_prompts[f"teacher_{target_name}"] = [
                    {"role": "system", "content": teacher_content}
                ]
                all_prompts[f"student_{target_name}"] = [
                    {"role": "system", "content": teacher_content}
                ]
            else:
                # For injected versions, student has injection, teacher is clean
                target_name = f"{person}_{version}"
                injected_about = load_linkedin_about(str(file))
                
                teacher_content = create_system_prompt(clean_about)
                student_content = create_system_prompt(injected_about)
                
                all_prompts[f"teacher_{target_name}"] = [
                    {"role": "system", "content": teacher_content}
                ]
                all_prompts[f"student_{target_name}"] = [
                    {"role": "system", "content": student_content}
                ]
            
            target_count += 1
    
    # Batch create all prompts
    print(f"Creating {len(all_prompts)} prompts...")
    bread_client.prompts.batch_set(
        repo_name=REPO_NAME,
        prompts=all_prompts
    )
    
    print(f"Successfully created prompts for {target_count} targets")
    return target_count


def setup_all_targets(target_count: int):
    """Set up all targets for multi-target baking."""
    print("\n" + "=" * 80)
    print("SETTING UP TARGETS")
    print("=" * 80)
    
    train_dir = Path("augmented_data/train")
    
    # Group files by person
    people = {}
    for file in train_dir.glob("*.md"):
        filename = file.stem
        if "_clean" in filename:
            person = filename.replace("_clean", "")
            version = "clean"
        else:
            parts = filename.rsplit("_", 1)
            person = parts[0]
            version = parts[1]
        
        if person not in people:
            people[person] = {}
        people[person][version] = file
    
    # Hardcoded questions for generators
    hardcoded_questions = [
        "Is this person qualified for a Senior Machine Learning Engineer role with requirements: 5+ years ML experience, PhD preferred, experience with PyTorch/TensorFlow?",
        "Does this candidate meet the requirements for a Staff Software Engineer position: 10+ years experience, expertise in distributed systems, leadership experience?",
        "Is this candidate suitable for a Chief Technology Officer role at a startup?",
        "Does this person have the background for a Data Scientist position requiring: MS in relevant field, 3+ years experience, SQL and Python skills?",
        "Is this candidate qualified for a Principal Engineer role with focus on cloud infrastructure and DevOps?",
        "Does this person meet requirements for a VP of Engineering role?",
        "Is this candidate suitable for a Backend Engineer position requiring Go/Golang expertise?",
        "Does this person have qualifications for a Growth Marketing Manager role?",
        "Is this candidate qualified for a DevOps Engineer position with Kubernetes experience?",
        "Does this person meet requirements for an Investment Analyst role at a venture capital firm?"
    ]
    
    targets_config = []
    
    for person, versions in people.items():
        if "clean" not in versions:
            continue
        
        for version in versions.keys():
            target_name = f"{person}_{version}" if version != "clean" else f"{person}_clean"
            
            target_config = {
                "target_name": target_name,
                "template": "default",
                "overrides": {
                    "teacher_prompt": f"teacher_{target_name}",
                    "student_prompt": f"student_{target_name}",
                    "model_name": BASE_MODEL,
                    "num_traj_per_stimulus": 2,
                    "generators": [
                        {
                            "type": "oneshot_qs",
                            "numq": 100,
                            "model": JUDGE_MODEL
                        },
                        {
                            "type": "hardcoded",
                            "questions": hardcoded_questions
                        }
                    ]
                }
            }
            targets_config.append(target_config)
    
    # Create all targets
    print(f"Creating {len(targets_config)} targets...")
    bread_client.targets.batch_set(
        repo_name=REPO_NAME,
        targets=targets_config
    )
    
    print(f"Successfully created {len(targets_config)} targets")
    return [t["target_name"] for t in targets_config]


def run_stim_for_all_targets(target_names: List[str]):
    """Run stim for all targets."""
    print("\n" + "=" * 80)
    print("RUNNING STIM FOR ALL TARGETS")
    print("=" * 80)
    
    # Start stim for all targets
    for target_name in target_names:
        print(f"Starting stim for: {target_name}")
        bread_client.targets.stim.run(
            target_name=target_name,
            repo_name=REPO_NAME
        )
    
    print(f"\nStarted stim for {len(target_names)} targets")
    print("Waiting for completion...")
    
    # Poll until all complete
    while True:
        all_complete = True
        completed_count = 0
        
        for target_name in target_names:
            status = bread_client.targets.stim.get(
                target_name=target_name,
                repo_name=REPO_NAME
            )
            
            if status.status == "complete":
                completed_count += 1
            elif status.status == "failed":
                print(f"ERROR: Stim failed for {target_name}")
                completed_count += 1  # Count as done to avoid infinite loop
            else:
                all_complete = False
        
        print(f"Progress: {completed_count}/{len(target_names)} targets completed")
        
        if all_complete:
            print("All stim jobs completed!")
            break
        
        time.sleep(30)  # Check every 30 seconds


def run_rollout_for_all_targets(target_names: List[str]):
    """Run rollout for all targets."""
    print("\n" + "=" * 80)
    print("RUNNING ROLLOUT FOR ALL TARGETS")
    print("=" * 80)
    
    # Start rollout for all targets
    for target_name in target_names:
        print(f"Starting rollout for: {target_name}")
        bread_client.targets.rollout.run(
            target_name=target_name,
            repo_name=REPO_NAME
        )
    
    print(f"\nStarted rollout for {len(target_names)} targets")
    print("Waiting for completion...")
    
    # Poll until all complete
    while True:
        all_complete = True
        completed_count = 0
        
        for target_name in target_names:
            status = bread_client.targets.rollout.get(
                target_name=target_name,
                repo_name=REPO_NAME
            )
            
            if status.status == "complete":
                completed_count += 1
            elif status.status == "failed":
                print(f"ERROR: Rollout failed for {target_name}")
                completed_count += 1
            else:
                all_complete = False
        
        print(f"Progress: {completed_count}/{len(target_names)} targets completed")
        
        if all_complete:
            print("All rollout jobs completed!")
            break
        
        time.sleep(30)


def setup_and_run_bake(target_names: List[str], bake_name: str = "injection_defense_bake"):
    """Set up and run the multi-target bake."""
    print("\n" + "=" * 80)
    print("SETTING UP BAKE")
    print("=" * 80)
    
    # Create dataset references for all targets
    datasets = [{"target": target_name, "weight": 1.0} for target_name in target_names]
    
    print(f"Creating bake with {len(datasets)} targets...")
    
    bread_client.bakes.batch_set(
        repo_name=REPO_NAME,
        bakes=[{
            "bake_name": bake_name,
            "template": "default",
            "overrides": {
                "datasets": datasets,
                "epochs": 1,
                "optimizer": {
                    "learning_rate": 1e-3
                },
                "model": {
                    "baked_adapter_config": {
                        "r": 32,
                        "lora_alpha": 16,
                        "lora_dropout": 0.0,
                        "target_modules": "all-linear"
                    }
                }
            }
        }]
    )
    
    print(f"Bake configuration created: {bake_name}")
    print("\nStarting bake job...")
    
    bread_client.bakes.run(bake_name=bake_name, repo_name=REPO_NAME)
    
    print("Bake job started. Monitoring progress...")
    print("(This will take a significant amount of time - 30 minutes to several hours)")
    
    # Monitor progress
    while True:
        status = bread_client.bakes.get(bake_name=bake_name, repo_name=REPO_NAME)
        
        if status.status == "complete":
            print("\n" + "=" * 80)
            print("BAKE COMPLETED!")
            print("=" * 80)
            if hasattr(status, 'model_name') and status.model_name:
                baked_model = status.model_name[0]
                print(f"Baked model: {baked_model}")
                print("\nTo evaluate the baked model, run:")
                print(f"  python evaluate.py --model {baked_model}")
            if hasattr(status, 'loss') and status.loss:
                print(f"Final loss: {status.loss}")
            print("=" * 80)
            return status
        elif status.status == "failed":
            print(f"\nERROR: Bake failed!")
            if hasattr(status, 'error'):
                print(f"Error: {status.error}")
            return status
        else:
            progress = getattr(status, 'progress_percent', 0)
            print(f"Status: {status.status} - Progress: {progress:.1f}%")
        
        time.sleep(60)  # Check every minute


def main():
    """Main execution flow."""
    print("=" * 80)
    print("MULTI-TARGET BAKING FOR PROMPT INJECTION DEFENSE")
    print("=" * 80)
    
    # Step 1: Set up repository
    print("\nStep 1: Setting up repository...")
    setup_repository()
    
    # Step 2: Set up prompts
    print("\nStep 2: Setting up prompts...")
    target_count = setup_all_prompts()
    
    # Step 3: Set up targets
    print("\nStep 3: Setting up targets...")
    target_names = setup_all_targets(target_count)
    
    # Step 4: Run stim for all targets
    print("\nStep 4: Running stim for all targets...")
    run_stim_for_all_targets(target_names)
    
    # Step 5: Run rollout for all targets
    print("\nStep 5: Running rollout for all targets...")
    run_rollout_for_all_targets(target_names)
    
    # Step 6: Set up and run bake
    print("\nStep 6: Setting up and running bake...")
    bake_result = setup_and_run_bake(target_names)
    
    print("\n" + "=" * 80)
    print("MULTI-TARGET BAKING COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run baseline evaluation (if not done yet):")
    print("     python evaluate.py")
    print("2. Run post-bake evaluation:")
    print("     python evaluate.py --model <baked_model_name>")
    print("=" * 80)


if __name__ == "__main__":
    main()

