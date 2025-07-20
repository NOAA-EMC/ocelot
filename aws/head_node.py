from run_cmd import run_cmd

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
            f"pcluster ssh --cluster-name {self.name} -i {self.key_file} -- \"{cmd};\";"
        )

        return run_cmd(cmd)

    def srun(self, script_path: str, num_nodes: int = 1, cpus_per_task: int = 4, path: str = None):
        """
        Runs a Python script on the head node of the AWS ParallelCluster.

        Parameters:
            script_path (str): Path to the Python script to run.
        """
        cmd = ""
        if path is not None:
            cmd += f"cd {path}; "
        cmd += f"srun --nodes {num_nodes} --cpus-per-task {cpus_per_task} (source ~/venv/bin/activate && python3.10 {script_path})"

        # Run the command on the head node
        self.run(cmd)

    def _initialize_python_env(self):
        """
        Initializes the Python virtual environment for Ocelot.
        """

        # Check if the .env directory exists on the cluster head node

        cmd = '''
        if [ ! -d "venv" ]; then
            echo "Creating Python virtual environment..."

            python3.10 -m venv venv

            python3.10 -m pip install --no-input --upgrade pip

            python3.10 -m pip install --no-input numpy==1.26.4
            python3.10 -m pip install --no-input pandas==2.2.2
            python3.10 -m pip install --no-input torch==2.5.1
            python3.10 -m pip install --no-input torch_scatter -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
            python3.10 -m pip install --no-input torch-geometric==2.6.1
            python3.10 -m pip install --no-input lightning==2.5.1
            python3.10 -m pip install --no-input scikit-learn==1.6.1
            python3.10 -m pip install --no-input matplotlib==3.9.4
            python3.10 -m pip install --no-input psutil==5.9.8
            python3.10 -m pip install --no-input trimesh==4.6.10
            python3.10 -m pip install --no-input zarr==2.18.0
        fi

        '''

        # Run the command on the head node
        self.run(cmd)

