import glob
import os


def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    """Find the most recent checkpoint file"""
    if not os.path.exists(checkpoint_dir):
        return None

    # Look for last.ckpt variants (these represent latest training state)
    last_pattern = os.path.join(checkpoint_dir, "last*.ckpt")
    last_checkpoints = glob.glob(last_pattern)

    if last_checkpoints:
        # Return the most recent "last" checkpoint by modification time
        latest_last = max(last_checkpoints, key=os.path.getmtime)
        return latest_last

    # if no "last" checkpoint exists, start fresh
    return None
