import os
import subprocess
import tempfile
import time
import yaml

import settings
from head_node import HeadNode
from run_cmd import run_cmd
from log import get_logger


logger = get_logger(__name__)

class Cluster:
    def __init__(self, name: str, instance_type: str, num_compute_nodes: int, cpus_per_task: int):
        self.name = name
        self.instance_type = instance_type
        self.num_compute_nodes = num_compute_nodes
        self.head_node = None
        self.config_path = None

        self._create()

    def __repr__(self):
        return f"Cluster(name={self.name}, region={settings.REGION})"

    def __str__(self):
        return f"Cluster {self.name} in {settings.REGION}"

    def __del__(self):
        if self.config_path:
            os.unlink(self.config_path)

    def delete(self):
        """
        Deletes the AWS ParallelCluster.
        """
        logger.info("Deleting Cluster '%s'", self.name)

        # Run the command to delete the cluster
        cmd = f"pcluster delete-cluster --cluster-name {self.name}"
        run_cmd(cmd)

        logger.info("Cluster '%s' deleted successfully.", self.name)
        self.head_node = None

    def _create(self) -> None:
        """
        Creates an AWS ParallelCluster with the specified instance type and number of compute nodes.
        """

        result = subprocess.run(
            ["pcluster", "describe-cluster", "--cluster-name", self.name],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            logger.info("Cluster '%s' already exists.", self.name)

        else:
            logger.info("Creating Cluster '%s'", self.name)

            # Prepare the configuration file
            self.config_path = self._prepare_config(self.instance_type, self.num_compute_nodes)
            logger.info("Cluster configuration file created at: %s", self.config_path)

            # Create the cluster using AWS ParallelCluster CLI
            cmd = f"pcluster create-cluster --cluster-name {self.name} " \
                  f"--cluster-configuration {self.config_path} --rollback-on-failure false"

            run_cmd(cmd)

        self._wait_for_cluster_creation()
        self.head_node = HeadNode(self.name, settings.KEY_FILE)


    def _wait_for_cluster_creation(self) -> None:
        """
        Blocking method to wait until the cluster is created.
        """

        while True:
            # Describe the cluster to get its status
            result = subprocess.run(["pcluster", "describe-cluster", "--cluster-name", self.name],
                                    capture_output=True, text=True, check=True)
            output = result.stdout

            # Parse the output to find the cluster status
            status_line = next((line for line in output.splitlines() if "clusterStatus" in line), None)
            if status_line:
                status = status_line.split(":")[1].strip().replace('"', '').replace(',', '')

                # Check if the status is CREATE_COMPLETE
                if status == "CREATE_COMPLETE":
                    logger.info("Cluster '%s' Creation Complete.", self.name)
                    break
                elif status == "CREATE_FAILED":
                    logger.error("Cluster '%s' Creation Failed.", self.name)
                    # Handle failed creation, e.g., print error message, exit script
                    logger.error(output)
                    break
                else:
                    logger.info("Cluster '%s' status is %s. Waiting...", self.name, status)

            # Wait before checking again
            time.sleep(30)  # Poll every 30 seconds

    def _prepare_config(self, instance_type: str, num_compute_nodes: int) -> str:

        # Load the template configuration file
        template_path = os.path.join(os.path.split(__file__)[0], 'cluster-config.yaml')
        with open(template_path) as f:
            config = yaml.safe_load(f)

        # Update the configuration with local settings
        config['Region'] = settings.REGION
        # config['Image']['Os'] = settings.OS
        config['Iam']['Roles']['LambdaFunctionsRole'] = settings.ADMIN_ROLE
        config['HeadNode']['InstanceType'] = settings.HEAD_NODE_INSTANCE
        config['HeadNode']['Iam']['InstanceRole'] = settings.HEAD_NODE_ROLE
        config['HeadNode']['Networking']['SubnetId'] = settings.SUBNET_ID
        config['HeadNode']['Ssh']['KeyName'] = settings.KEY_NAME

        for queue in config['Scheduling']['SlurmQueues']:
            queue['Networking']['SubnetIds'] = [settings.SUBNET_ID]
            queue['Iam']['InstanceRole'] = settings.HEAD_NODE_ROLE
            queue['CustomActions']['OnNodeStart']['Script'] = settings.ON_NODE_START_SCRIPT
            for cr in queue['ComputeResources']:
                cr['InstanceType'] = instance_type
                cr['MinCount'] = num_compute_nodes
                cr['MaxCount'] = num_compute_nodes
        for storage in config.get('SharedStorage', []):
            if storage.get('StorageType') == 'FsxLustre':
                lustre_settings = storage.get('FsxLustreSettings', {})
                lustre_settings['ImportPath'] = settings.IMPORT_PATH
                lustre_settings['ExportPath'] = f'{settings.EXPORT_PATH}/{self.name}'

        with tempfile.NamedTemporaryFile(
                mode='w',  # text mode
                encoding='utf-8',  # explicit for clarity
                delete=False,
                suffix='.yaml') as tmp:

            yaml.safe_dump(config, tmp, sort_keys=False)  # stream is now text
            return tmp.name  # tmp is already closed by the context manager
