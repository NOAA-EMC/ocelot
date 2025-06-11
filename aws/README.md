# AWS Training Scripts

This directory contains helper files for running the GNN model in
`gnn_model/train_gnn.py` on ephemeral AWS ParallelCluster resources.

The typical workflow is:

1. Prepare a **S3 bucket** with your training data. Results will also be
   written back to this bucket.
2. Configure the `cluster-config.yaml` file with your VPC and subnet IDs.
3. Launch the workflow using `run_training.py`. The script will
   create a cluster, copy the repository to the head node, run the
   training script and tear everything down.

```
python run_training.py --key-name <ssh-key> --s3-bucket <bucket> \
    --instance-type g5.2xlarge
```

The script requires the AWS ParallelCluster CLI to be installed and configured
with credentials that can create and manage the cluster.
