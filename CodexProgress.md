# Codex Progress Log

- 2026-04-22: Started work on reducing future checkpoint sizes and adding an append-only progress log.
- 2026-04-22: Confirmed `CodexProgress.md` did not exist and inspected `code/train.py`.
- 2026-04-22: Identified the current checkpoint behavior in `code/train.py`: each save writes both an epoch snapshot and `checkpoint_latest.pth.tar`, and each checkpoint includes model weights plus optimizer, auxiliary optimizer, and scheduler state.
- 2026-04-22: Patched `code/train.py` so checkpoint saving is leaner by default.
- 2026-04-22: Added `unwrap_model()` and `build_checkpoint_payload()` helpers in `code/train.py`.
- 2026-04-22: Changed checkpoint payload construction so weights-only checkpoints are the default; optimizer, auxiliary optimizer, and scheduler state are only included when `--save-training-state` is passed.
- 2026-04-22: Changed checkpoint file retention so only `checkpoint_latest.pth.tar` is kept by default; per-epoch snapshots are only written when `--save-epoch-checkpoints` is passed.
- 2026-04-22: Kept training resume compatibility in `code/train.py` by making optimizer / scheduler restoration conditional and printing explicit messages when loading a weights-only checkpoint.
- 2026-04-22: Verified the edited Python files compile successfully after the checkpointing patch.
- 2026-04-22: Ran a save-size probe in the workspace using the smaller smoke-test model configuration.
- 2026-04-22: Measured the new default checkpoint behavior at approximately 39.5 MB for a weights-only `checkpoint_latest.pth.tar`.
- 2026-04-22: Measured the opt-in full-state behavior at approximately 118.6 MB for `checkpoint_latest.pth.tar`, with an additional approximately 118.5 MB per-epoch snapshot when `--save-epoch-checkpoints` is enabled.
- 2026-04-22: Observed a OneDrive / Windows reparse-point cleanup issue while deleting the temporary `tmp_checkpoint_probe` directory after verification; the probe files remained in place after attempted removal.
- 2026-04-22: Began preparing the current CATC checkpoint-size changes for commit and push to the user's GitHub repository.
- 2026-04-22: Checked the git status, confirmed the active remote, and confirmed there was no existing `.gitignore` file in the repository root.
- 2026-04-22: Added a minimal `.gitignore` to keep local Python bytecode artifacts (`__pycache__` and `*.py[cod]`) out of the commit while pushing the requested source changes.
- 2026-04-22: User approved proceeding with staging, committing, and pushing the current checkpoint-size cleanup changes to GitHub using the requested commit message.
