import argparse
import os
import subprocess
import tempfile

import yaml


def run_command(cmd: str):
    print(cmd)
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def prepare_config(template_path: str, instance_type: str, key_name: str, s3_bucket: str) -> str:
    with open(template_path) as f:
        config = yaml.safe_load(f)

    config['HeadNode']['Ssh']['KeyName'] = key_name
    for queue in config['Scheduling']['SlurmQueues']:
        for cr in queue['ComputeResources']:
            cr['InstanceType'] = instance_type
    for storage in config.get('SharedStorage', []):
        if storage.get('StorageType') == 'FsxLustre':
            settings = storage.get('FsxLustreSettings', {})
            if 'ImportPath' in settings:
                settings['ImportPath'] = f"s3://{s3_bucket}/training-data"
            if 'ExportPath' in settings:
                settings['ExportPath'] = f"s3://{s3_bucket}/output"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.yaml')
    yaml.safe_dump(config, tmp)
    tmp.close()
    return tmp.name


def main():
    parser = argparse.ArgumentParser(description='Run ocelot training on AWS ParallelCluster')
    parser.add_argument('--cluster-name', default='ocelot-training', help='Name of the cluster')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--config', default='cluster-config.yaml', help='Cluster configuration template')
    parser.add_argument('--instance-type', default='g5.2xlarge', help='EC2 instance type for compute nodes')
    parser.add_argument('--key-name', required=True, help='EC2 key pair name for SSH access')
    parser.add_argument('--training-script', default='gnn_model/train_gnn.py', help='Path to training script')
    parser.add_argument('--s3-bucket', required=True, help='S3 bucket used with FSx Lustre')
    parser.add_argument('--keep-cluster', action='store_true', help='Do not delete the cluster after completion')
    args = parser.parse_args()

    cfg_path = prepare_config(args.config, args.instance_type, args.key_name, args.s3_bucket)

    create_cmd = (
        f"pcluster create-cluster --cluster-name {args.cluster_name} "
        f"--region {args.region} --cluster-configuration {cfg_path} "
        "--rollback-on-failure false"
    )
    run_command(create_cmd)

    wait_cmd = (
        f"pcluster wait cluster-available --cluster-name {args.cluster_name} "
        f"--region {args.region}"
    )
    run_command(wait_cmd)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    scp_cmd = (
        f"pcluster scp --cluster-name {args.cluster_name} --region {args.region} "
        f"--recursive {repo_root} headnode:/home/ec2-user/ocelot"
    )
    run_command(scp_cmd)

    remote_cmd = (
        "cd ocelot && "
        "pip install -r gnn_model/requirements.txt && "
        f"python {args.training_script}"
    )
    ssh_cmd = (
        f"pcluster ssh --cluster-name {args.cluster_name} --region {args.region} "
        f"--command \"{remote_cmd}\""
    )
    run_command(ssh_cmd)

    if not args.keep_cluster:
        run_command(f"pcluster delete-cluster --cluster-name {args.cluster_name} --region {args.region} --yes")

    os.unlink(cfg_path)


if __name__ == '__main__':
    main()
