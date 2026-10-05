from huggingface_hub import snapshot_download

# Weights and prediction configurations of Task 2, written to ./Task_2/ (can be run again to update them).
snapshot_download(repo_id="VBoussot/Synthrad2025", allow_patterns="Task_2/*", local_dir=".")
