import argparse
import os
import subprocess
import settings
from cluster import Cluster

def init_ocelot_branch(cluster: Cluster, branch: str = 'main'):
    """
    Initializes the Ocelot branch on the cluster.
    :param cluster: The Cluster object representing the AWS ParallelCluster.
    :param branch: The branch of Ocelot to initialize.
    """
    remote_cmd = f'''
    if [ ! -d ocelot ]; then 
        git clone https://github.com/NOAA-EMC/ocelot.git
    fi
        
    cd ocelot
    git checkout {branch}
    '''

    cluster.head_node.run(remote_cmd)

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

    cluster = Cluster(args.cluster_name, args.instance_type, args.num_nodes, args.cpus_per_task)
    init_ocelot_branch(cluster, args.branch)

    cluster.head_node.srun(args.script, num_nodes=args.num_nodes, cpus_per_task=args.cpus_per_task, path="ocelot")

    if not args.keep_cluster:
        cluster.delete()

if __name__ == '__main__':
    main()
