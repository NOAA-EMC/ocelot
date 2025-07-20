import os
import subprocess

def run_cmd(cmd: str) -> str:
    """
    Runs a command in the terminal and returns the output.
    :param cmd:
    """

    result = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,  # so stdout/stderr are str, not bytes
        env=os.environ  # inherit your AWS credentials, PATH, etc.
    )

    if result.returncode != 0:
        raise RuntimeError(f"{cmd} failed with error: {result.stderr.strip()}")

    return result.stdout.strip()
