import argparse
import os
import subprocess
import tempfile

import yaml


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



def prepare_config(instance_type: str, data_source: str, output_path: str) -> str:

    template_path = './cluster-config.yaml'
    with open(template_path) as f:
        config = yaml.safe_load(f)

    for queue in config['Scheduling']['SlurmQueues']:
        for cr in queue['ComputeResources']:
            cr['InstanceType'] = instance_type
    for storage in config.get('SharedStorage', []):
        if storage.get('StorageType') == 'FsxLustre':
            settings = storage.get('FsxLustreSettings', {})
            if 'ImportPath' in settings:
                settings['ImportPath'] = f"s3://noaa-ocelot/{data_source}"
            if 'ExportPath' in settings:
                settings['ExportPath'] = f"s3://noaa-ocelot/{output_path}"

    with tempfile.NamedTemporaryFile(
            mode='w',            # text mode
            encoding='utf-8',    # explicit for clarity
            delete=False,
            suffix='.yaml') as tmp:
        yaml.safe_dump(config, tmp, sort_keys=False)   # <‑‑ stream is now text
        return tmp.name        # tmp is already closed by the context manager


def main():
    parser = argparse.ArgumentParser(description='Run ocelot training on AWS ParallelCluster')
    parser.add_argument('--cluster-name', default='ocelot-training', help='Name of the cluster')
    parser.add_argument('--instance-type', default='g5.2xlarge', help='EC2 instance type for compute nodes')
    parser.add_argument('--training-script', default='gnn_model/train_gnn.py', help='Path to training script')
    parser.add_argument('--data-source', required=True, help='The data source path to use in the s3 bucket')
    parser.add_argument('--output', required=True, help='Output path in the s3 bucket.')
    parser.add_argument('--keep-cluster', action='store_true', help='Do not delete the cluster after completion')
    args = parser.parse_args()

    cfg_path = prepare_config(args.instance_type, args.data_source, args.output)

    region = "us-east-1"

    create_cmd = (
        f"pcluster create-cluster --cluster-name {args.cluster_name} "
        f"--region {region} --cluster-configuration {cfg_path} "
        "--rollback-on-failure false"
    )
    run_command(create_cmd)

    wait_cmd = (
        f"pcluster wait cluster-available --cluster-name {args.cluster_name} "
        f"--region {region}"
    )
    run_command(wait_cmd)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    scp_cmd = (
        f"pcluster scp --cluster-name {args.cluster_name} --region {region} "
        f"--recursive {repo_root} headnode:/home/ubuntu/ocelot"
    )
    run_command(scp_cmd)

    remote_cmd = (
        "cd ocelot && "
        "pip install -r gnn_model/requirements.txt && "
        f"python {args.training_script}"
    )
    ssh_cmd = (
        f"pcluster ssh --cluster-name {args.cluster_name} --region {region} "
        f"--command \"{remote_cmd}\""
    )
    run_command(ssh_cmd)

    if not args.keep_cluster:
        run_command(f"pcluster delete-cluster --cluster-name {args.cluster_name} --region {args.region} --yes")

    os.unlink(cfg_path)


if __name__ == '__main__':
    main()
