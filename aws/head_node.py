from log import get_logger
from run_cmd import run_cmd

logger = get_logger(__name__)

class HeadNode:
    """
    Represents the head node in a aws cluster.
    """

    def __init__(self, name: str, key_file: str):
        """
        Initializes the HeadNode with a name and IP address.

        :param name: The name of the head node.
        :param key_file: The path to the private key file.
        """
        self.name = name
        self.key_file = key_file
        self._initialize_python_env()

    def __repr__(self):
        return f"HeadNode(name={self.name})"

    def run(self, cmd: str) -> str:
        """
        Runs a command on the head node of the AWS ParallelCluster.
        """
        cmd = (
            f"pcluster ssh --cluster-name {self.name} -i {self.key_file} -- bash -lc \"\\\"{cmd};\\\"\";"
        )

        logger.info("Cluster %s - Running command: %s", self.name, cmd)
        result = run_cmd(cmd)
        logger.info("Cluster %s - Command executed with result: %s", self.name, result)

        return result

    def srun(self, script_path: str, num_nodes: int = 1, cpus_per_task: int = 4, working_dir: str = None):
        """
        Runs a Python script on the head node of the AWS ParallelCluster.

        Parameters:
            script_path (str): Path to the Python script to run.
        """
        cmd = ""
        if working_dir is not None:
            cmd += f"cd {working_dir}; "
        cmd += f"srun --nodes {num_nodes} --cpus-per-task {cpus_per_task} python3.10 {script_path}"

        # Run the command on the head node
        self.run(cmd)

    def _initialize_python_env(self):
        """
        Initializes the Python virtual environment for Ocelot.
        """

        # Check if the .env directory exists on the cluster head node

        logger.info("Cluster %s - Initializing Python environment...", self.name)

        cmd = "/fsx/input/scripts/init_python_env.sh"
        result = self.run(cmd)

        logger.info("Cluster %s - Python environment initialized with result %s.", self.name, result)

