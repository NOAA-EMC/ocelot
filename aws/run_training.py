import argparse
import os
import subprocess
import tempfile

import time
import yaml

# Added configuration from local settings
try:
    import local_settings as settings
except ImportError:
    raise RuntimeError(f"local_settings not found. Please install local_settings first.")


#def run_command(cmd: str):
#    print(cmd)
#    result = subprocess.run(cmd, shell=True)
#    if result.returncode != 0:
#        raise RuntimeError(f"Command failed: {cmd}")

def run_command(cmd: str):
    print(f">>> {cmd}")
    # capture everything and decode to str
    result = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,            # so stdout/stderr are str, not bytes
        env=os.environ        # inherit your AWS credentials, PATH, etc.
    )
    print("---- STDOUT ----")
    print(result.stdout.strip())
    print("---- STDERR ----")
    print(result.stderr.strip())
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {cmd}")


def wait_for_cluster_creation(cluster_name):
    """
    Waits for AWS ParallelCluster status to become CREATE_COMPLETE.
    """
    while True:
        # Describe the cluster to get its status
        result = subprocess.run(["pcluster", "describe-cluster", "--cluster-name", cluster_name], capture_output=True, text=True, check=True)
        output = result.stdout

        # Parse the output to find the cluster status
        status_line = next((line for line in output.splitlines() if "clusterStatus" in line), None)
        if status_line:
            status = status_line.split(":")[1].strip().replace('"', '').replace(',', '')

            # Check if the status is CREATE_COMPLETE
            if status == "CREATE_COMPLETE":
                print(f"Cluster '{cluster_name}' creation complete.")
                break
            elif status == "CREATE_FAILED":
                print(f"Cluster '{cluster_name}' creation failed.")
                # Handle failed creation, e.g., print error message, exit script
                print(output)
                break
            else:
                print(f"Cluster '{cluster_name}' status is {status}. Waiting...")

        # Wait before checking again
        time.sleep(30) # Poll every 30 seconds


def prepare_config(instance_type: str, num_compute_nodes: int) -> str:
    # Validate local settings
    setting_items = ["OS",
                     "REGION",
                     "SUBNET_ID",
                     "ADMIN_ROLE",
                     "HEAD_NODE_INSTANCE",
                     "HEAD_NODE_ROLE",
                     "COMPUTE_NODE_ROLE",
                     "IMPORT_PATH",
                     "EXPORT_PATH",
                     "KEY_FILE",
                     "ON_NODE_START_SCRIPT",
                     "ON_NODE_CONFIGURED_SCRIPT"]
    for item in setting_items:
        if not hasattr(settings, item):
            raise RuntimeError(f"local_settings is missing required setting: {item}")


    # Load the template configuration file
    template_path = './cluster-config.yaml'
    with open(template_path) as f:
        config = yaml.safe_load(f)

    # Update the configuration with local settings
    config['Region'] = settings.REGION
    config['Image']['Os'] = settings.OS
    config['Iam']['Roles']['LambdaFunctionsRole'] = settings.ADMIN_ROLE
    config['HeadNode']['InstanceType'] = settings.HEAD_NODE_INSTANCE
    config['HeadNode']['Iam']['InstanceRole'] = settings.HEAD_NODE_ROLE
    config['HeadNode']['Networking']['SubnetId'] = settings.SUBNET_ID

    for queue in config['Scheduling']['SlurmQueues']:
        queue['Networking']['SubnetIds'] = [settings.SUBNET_ID]
        queue['Iam']['InstanceRole'] = settings.HEAD_NODE_ROLE
        queue['CustomActions']['OnNodeStart']['Script'] = settings.ON_NODE_START_SCRIPT
        queue['CustomActions']['OnNodeConfigured']['Script'] = settings.ON_NODE_CONFIGURED_SCRIPT
        for cr in queue['ComputeResources']:
            cr['InstanceType'] = instance_type
            cr['MinCount'] = num_compute_nodes
            cr['MaxCount'] = num_compute_nodes
    for storage in config.get('SharedStorage', []):
        if storage.get('StorageType') == 'FsxLustre':
            lustre_settings = storage.get('FsxLustreSettings', {})
            lustre_settings['ImportPath'] = settings.IMPORT_PATH
            lustre_settings['ExportPath'] = settings.EXPORT_PATH

    with tempfile.NamedTemporaryFile(
            mode='w',            # text mode
            encoding='utf-8',    # explicit for clarity
            delete=False,
            suffix='.yaml') as tmp:
        
        yaml.safe_dump(config, tmp, sort_keys=False)   # stream is now text
        return tmp.name        # tmp is already closed by the context manager


def main():
    parser = argparse.ArgumentParser(description='Run ocelot training on AWS ParallelCluster')
    parser.add_argument('--cluster-name', default='ocelot-training', help='Name of the cluster')
    parser.add_argument('--branch', default='main', help='Ocelot branch to use')
    parser.add_argument('--instance-type', default='g5.2xlarge', help='EC2 instance type for compute nodes')
    parser.add_argument('--script', default='gnn_model/train_gnn.py', help='Python script to run on the cluster')
    parser.add_argument('--num_nodes', type=int, help='Number of nodes in the cluster.')
    parser.add_argument('--cpus_per_task', type=int, default=4, help='Number of CPUs per node')
    parser.add_argument('--keep-cluster', action='store_true', help='Do not delete the cluster after completion')
    args = parser.parse_args()

    cfg_path = prepare_config(args.instance_type, args.num_nodes)
    print ('!!!!', cfg_path)
    result = subprocess.run(
       ["pcluster", "describe-cluster", "--cluster-name", args.cluster_name],
       capture_output=True, text=True
    )

    if result.returncode == 0:
        print(f"Cluster '{args.cluster_name}' already exists.")

    else:
        print(f"Creating Cluster '{args.cluster_name}'")
        create_cmd = (
                f"pcluster create-cluster --cluster-name {args.cluster_name} "
                f"--cluster-configuration {cfg_path} --rollback-on-failure false"
            )
        run_command(create_cmd)

    wait_for_cluster_creation(args.cluster_name)


    remote_cmd = (
        "cd ocelot && git checkout " + args.branch + " && " + "source venv/bin/activate " + "&& "
        f"srun --nodes {args.num_nodes} --cpus-per-task {args.cpus_per_task} python3 {args.script} "
    )

    ssh_cmd = (
        f"pcluster ssh --cluster-name {args.cluster_name} -i {settings.KEY_FILE} "
        f"-- \"{remote_cmd}\""
    )
    run_command(ssh_cmd)

    if not args.keep_cluster:
        run_command(f"pcluster delete-cluster --cluster-name {args.cluster_name} ")

    os.unlink(cfg_path)


if __name__ == '__main__':
    main()
