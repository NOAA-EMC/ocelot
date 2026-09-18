#!/bin/bash
#
# Script to submit multiple training experiments from a YAML configuration file
# Usage: bash submit_experiments_from_yaml.sh [config_file] [experiment_filter] [--debug] [--dry-run] [--version VERSION]
#
# Examples:
#   bash submit_experiments_from_yaml.sh                           # Submit all experiments
#   bash submit_experiments_from_yaml.sh experiment_configs.yaml   # Use specific config file
#   bash submit_experiments_from_yaml.sh - baseline                # Submit only experiments matching "baseline"
#   bash submit_experiments_from_yaml.sh - ablation                # Submit only ablation experiments
#   bash submit_experiments_from_yaml.sh - baseline --debug        # Submit baseline with 30-minute time limit
#   bash submit_experiments_from_yaml.sh - quick_train --dry-run    # Show what would be submitted without actually submitting
#   bash submit_experiments_from_yaml.sh - baseline --version v1    # Force logger version (logs/<exp_name>/v1)
#   bash submit_experiments_from_yaml.sh - test --dry-run   # YAML: trainer.test only (edit load_ckpt_path / data_path in YAML)
#   bash submit_experiments_from_yaml.sh - inference_predict --dry-run  # YAML inference: predict + save .pt
#   bash submit_experiments_from_yaml.sh - hier_ --dry-run   # Hierarchical mesh experiments (see experiment_configs.yaml)
#

# Check for flags
DEBUG_MODE=false
DRY_RUN=false
VERSION=""
POSITIONAL=()
while [ $# -gt 0 ]; do
    case "$1" in
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --version=*)
            VERSION="${1#*=}"
            shift
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL[@]}"

# Default configuration file
# If first arg is "-", use default config and treat second arg as filter
if [ "$1" = "-" ]; then
    CONFIG_FILE="experiment_configs.yaml"
    FILTER="${2:-}"
else
    CONFIG_FILE="${1:-experiment_configs.yaml}"
    FILTER="${2:-}"
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found!"
    exit 1
fi

# Base directory for job scripts
JOB_DIR="./job_scripts"
mkdir -p $JOB_DIR
mkdir -p jobs

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Python script to parse YAML and generate job scripts
PARSER_SCRIPT=$(cat << 'PYTHON_EOF'
import yaml
import sys
import os

def get_value(exp_config, defaults, key, required=False):
    """Get value from experiment config, falling back to defaults"""
    value = exp_config.get(key, defaults.get(key))
    if required and value is None:
        raise ValueError(f"Required parameter '{key}' not found in experiment or defaults")
    return value

def main():
    config_file = sys.argv[1]
    filter_str = sys.argv[2] if len(sys.argv) > 2 else ""
    debug_mode = sys.argv[3] == "true" if len(sys.argv) > 3 else False
    dry_run = sys.argv[4] == "true" if len(sys.argv) > 4 else False
    
    # Load YAML config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    defaults = config.get('defaults', {})
    experiments = config.get('experiments', {})
    
    if not experiments:
        print("No experiments found in configuration file!", file=sys.stderr)
        sys.exit(1)
    
    # Filter experiments
    if filter_str:
        experiments = {k: v for k, v in experiments.items() if filter_str in k}
        if not experiments:
            print(f"No experiments matching filter '{filter_str}'", file=sys.stderr)
            sys.exit(1)
    
    # Print to stderr so it doesn't interfere with data output
    print(f"Found {len(experiments)} experiment(s) to submit", file=sys.stderr)
    
    for exp_name, exp_config in experiments.items():
        if exp_config is None:
            exp_config = {}
        
        # Get all parameters with fallback to defaults
        nodes = get_value(exp_config, defaults, 'nodes', required=True)
        gpus_per_node = get_value(exp_config, defaults, 'gpus_per_node', required=True)
        time_limit = get_value(exp_config, defaults, 'time', required=True)
        partition = get_value(exp_config, defaults, 'partition', required=True)
        account = get_value(exp_config, defaults, 'account', required=True)
        
        # Model parameters
        hidden_dim = get_value(exp_config, defaults, 'hidden_dim', required=True)
        mesh_gnn_layers = get_value(exp_config, defaults, 'mesh_gnn_layers', required=True)
        num_processor_rollout_steps = get_value(exp_config, defaults, 'num_processor_rollout_steps')
        use_gat_encoder_decoder = get_value(exp_config, defaults, 'use_gat_encoder_decoder')
        num_heads = get_value(exp_config, defaults, 'num_heads')
        gat_chunk_size = get_value(exp_config, defaults, 'gat_chunk_size')
        hidden_layers = get_value(exp_config, defaults, 'hidden_layers')
        
        # Training parameters
        batch_size = get_value(exp_config, defaults, 'batch_size', required=True)
        lr = get_value(exp_config, defaults, 'lr', required=True)
        max_epochs = get_value(exp_config, defaults, 'max_epochs', required=True)
        validation_interval = get_value(exp_config, defaults, 'validation_interval')
        limit_val_batches = get_value(exp_config, defaults, 'limit_val_batches')
        num_workers = get_value(exp_config, defaults, 'num_workers')
        strategy = get_value(exp_config, defaults, 'strategy')
        precision = get_value(exp_config, defaults, 'precision')
        weight_decay = get_value(exp_config, defaults, 'weight_decay')
        loss = get_value(exp_config, defaults, 'loss')
        
        # Data parameters
        data_path = get_value(exp_config, defaults, 'data_path')
        start_date = get_value(exp_config, defaults, 'start_date')
        end_date = get_value(exp_config, defaults, 'end_date')
        delta_time = get_value(exp_config, defaults, 'delta_time')
        num_target_bins = get_value(exp_config, defaults, 'num_target_bins')
        no_obs_autoregressive = get_value(exp_config, defaults, 'no_obs_autoregressive')
        no_obs_ar_teacher_forcing = get_value(exp_config, defaults, 'no_obs_ar_teacher_forcing')
        normalize_increment = get_value(exp_config, defaults, 'normalize_increment')
        target_is_anal = get_value(exp_config, defaults, 'target_is_anal')
        gradient_loss_weight = get_value(exp_config, defaults, 'gradient_loss_weight')
        multiscale_loss_weight = get_value(exp_config, defaults, 'multiscale_loss_weight')
        multiscale_scales = get_value(exp_config, defaults, 'multiscale_scales')
        
        # Monitoring parameters
        monitor_memory = get_value(exp_config, defaults, 'monitor_memory')
        memory_log_interval = get_value(exp_config, defaults, 'memory_log_interval')
        profile = get_value(exp_config, defaults, 'profile')
        debug_gradients = get_value(exp_config, defaults, 'debug_gradients')
        
        # Inference-only (test / predict)
        action = get_value(exp_config, defaults, 'action')
        load_ckpt_path = get_value(exp_config, defaults, 'load_ckpt_path')
        predict_output = get_value(exp_config, defaults, 'predict_output')
        
        # Early stopping (train_hetero_model: --no_early_stopping, --early_stopping_patience)
        no_early_stopping = get_value(exp_config, defaults, 'no_early_stopping')
        early_stopping_patience = get_value(exp_config, defaults, 'early_stopping_patience')
        
        # Observation config variant
        obs_config = get_value(exp_config, defaults, 'obs_config')

        # Regional model parameters
        exp_type = get_value(exp_config, defaults, 'exp_type')
        model_path = get_value(exp_config, defaults, 'model_path')
        x_min = get_value(exp_config, defaults, 'x_min')
        x_max = get_value(exp_config, defaults, 'x_max')
        y_min = get_value(exp_config, defaults, 'y_min')
        y_max = get_value(exp_config, defaults, 'y_max')
        mesh_feature_dim = get_value(exp_config, defaults, 'mesh_feature_dim')
        cutoff_factor = get_value(exp_config, defaults, 'cutoff_factor')
        num_neighbors = get_value(exp_config, defaults, 'num_neighbors')

        # Hierarchical mesh + processor (train_hetero_model)
        hierarchical = get_value(exp_config, defaults, 'hierarchical')
        hierarchical_processor_type = get_value(exp_config, defaults, 'hierarchical_processor_type')
        mesh_splits = get_value(exp_config, defaults, 'mesh_splits')
        levels = get_value(exp_config, defaults, 'levels')
        processor_window = get_value(exp_config, defaults, 'processor_window')
        processor_depth = get_value(exp_config, defaults, 'processor_depth')
        processor_heads = get_value(exp_config, defaults, 'processor_heads')
        processor_dropout = get_value(exp_config, defaults, 'processor_dropout')
        spatial_mixing_steps = get_value(exp_config, defaults, 'spatial_mixing_steps')
        processor_attn_chunk_size = get_value(exp_config, defaults, 'processor_attn_chunk_size')
        no_processor_causal_mask = get_value(exp_config, defaults, 'no_processor_causal_mask')
        no_processor_cross_scale = get_value(exp_config, defaults, 'no_processor_cross_scale')
        
        description = exp_config.get('description', 'No description')
        
        # Override time in debug mode
        if debug_mode:
            time_limit = "00:30:00"
        
        # Output experiment info as shell variables
        print(f"EXP_NAME={exp_name}")
        print(f"DESCRIPTION='{description}'")
        print(f"NODES={nodes}")
        print(f"GPUS_PER_NODE={gpus_per_node}")
        print(f"TIME={time_limit}")
        print(f"PARTITION={partition}")
        print(f"ACCOUNT={account}")
        print(f"HIDDEN_DIM={hidden_dim}")
        print(f"MESH_GNN_LAYERS={mesh_gnn_layers}")
        print(f"NUM_PROCESSOR_ROLLOUT_STEPS={num_processor_rollout_steps if num_processor_rollout_steps is not None else ''}")
        print(f"USE_GAT_ENCODER_DECODER={str(use_gat_encoder_decoder).lower() if use_gat_encoder_decoder else 'false'}")
        print(f"NUM_HEADS={num_heads if num_heads else ''}")
        print(f"GAT_CHUNK_SIZE={gat_chunk_size if gat_chunk_size is not None else ''}")
        print(f"HIDDEN_LAYERS={hidden_layers if hidden_layers else ''}")
        print(f"BATCH_SIZE={batch_size}")
        print(f"LR={lr}")
        print(f"MAX_EPOCHS={max_epochs}")
        print(f"VALIDATION_INTERVAL={validation_interval if validation_interval else ''}")
        print(f"LIMIT_VAL_BATCHES={limit_val_batches if limit_val_batches else ''}")
        print(f"NUM_WORKERS={num_workers if num_workers else ''}")
        print(f"STRATEGY={strategy if strategy else ''}")
        print(f"PRECISION={precision if precision else ''}")
        print(f"WEIGHT_DECAY={weight_decay if weight_decay else ''}")
        print(f"LOSS={loss if loss else ''}")
        print(f"DATA_PATH={data_path if data_path else ''}")
        print(f"START_DATE={start_date if start_date else ''}")
        print(f"END_DATE={end_date if end_date else ''}")
        print(f"DELTA_TIME={delta_time if delta_time is not None else ''}")
        print(f"MONITOR_MEMORY={str(monitor_memory).lower() if monitor_memory else 'false'}")
        print(f"MEMORY_LOG_INTERVAL={memory_log_interval if memory_log_interval else ''}")
        print(f"PROFILE={str(profile).lower() if profile else 'false'}")
        print(f"DEBUG_GRADIENTS={'true' if debug_gradients else 'false'}")
        print(f"NUM_TARGET_BINS={num_target_bins if num_target_bins is not None else ''}")
        print(f"NO_OBS_AUTOREGRESSIVE={'true' if no_obs_autoregressive else 'false'}")
        print(f"NO_OBS_AR_TEACHER_FORCING={'true' if no_obs_ar_teacher_forcing else 'false'}")
        print(f"NORMALIZE_INCREMENT={'true' if normalize_increment else 'false'}")
        print(f"TARGET_IS_ANAL={'true' if target_is_anal else 'false'}")
        print(f"GRADIENT_LOSS_WEIGHT={gradient_loss_weight if gradient_loss_weight is not None else ''}")
        print(f"MULTISCALE_LOSS_WEIGHT={multiscale_loss_weight if multiscale_loss_weight is not None else ''}")
        print(f"MULTISCALE_SCALES={' '.join(str(s) for s in multiscale_scales) if multiscale_scales else ''}")
        print(f"ACTION={action if action else ''}")
        print(f"LOAD_CKPT_PATH={load_ckpt_path if load_ckpt_path else ''}")
        print(f"PREDICT_OUTPUT={predict_output if predict_output else ''}")
        print(f"NO_EARLY_STOPPING={'true' if no_early_stopping else 'false'}")
        print(f"EARLY_STOPPING_PATIENCE={early_stopping_patience if early_stopping_patience is not None else ''}")
        print(f"HIERARCHICAL={'true' if hierarchical else 'false'}")
        print(f"HIERARCHICAL_PROCESSOR_TYPE={hierarchical_processor_type if hierarchical_processor_type else ''}")
        print(f"MESH_SPLITS={mesh_splits if mesh_splits is not None else ''}")
        print(f"LEVELS={levels if levels is not None else ''}")
        print(f"PROCESSOR_WINDOW={processor_window if processor_window is not None else ''}")
        print(f"PROCESSOR_DEPTH={processor_depth if processor_depth is not None else ''}")
        print(f"PROCESSOR_HEADS={processor_heads if processor_heads is not None else ''}")
        print(f"PROCESSOR_DROPOUT={processor_dropout if processor_dropout is not None else ''}")
        print(f"SPATIAL_MIXING_STEPS={spatial_mixing_steps if spatial_mixing_steps is not None else ''}")
        print(f"PROCESSOR_ATTN_CHUNK_SIZE={processor_attn_chunk_size if processor_attn_chunk_size is not None else ''}")
        print(f"NO_PROCESSOR_CAUSAL_MASK={'true' if no_processor_causal_mask else 'false'}")
        print(f"NO_PROCESSOR_CROSS_SCALE={'true' if no_processor_cross_scale else 'false'}")
        print(f"OBS_CONFIG={obs_config if obs_config else ''}")
        print(f"EXP_TYPE={exp_type if exp_type else 'global'}")
        print(f"MODEL_PATH={model_path if model_path else ''}")
        print(f"X_MIN={x_min if x_min is not None else ''}")
        print(f"X_MAX={x_max if x_max is not None else ''}")
        print(f"Y_MIN={y_min if y_min is not None else ''}")
        print(f"Y_MAX={y_max if y_max is not None else ''}")
        print(f"MESH_FEATURE_DIM={mesh_feature_dim if mesh_feature_dim is not None else ''}")
        print(f"CUTOFF_FACTOR={cutoff_factor if cutoff_factor is not None else ''}")
        print(f"NUM_NEIGHBORS={num_neighbors if num_neighbors is not None else ''}")
        print("---EXPERIMENT_END---")

if __name__ == "__main__":
    main()
PYTHON_EOF
)

# Parse YAML and process experiments
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Experiment Submission Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Config file: ${YELLOW}$CONFIG_FILE${NC}"
if [ -n "$FILTER" ]; then
    echo -e "Filter: ${YELLOW}$FILTER${NC}"
fi
if [ -n "$VERSION" ]; then
    echo -e "Version: ${YELLOW}$VERSION${NC}"
fi
if [ "$DEBUG_MODE" = true ]; then
    echo -e "${YELLOW}DEBUG MODE: Time limit set to 30 minutes${NC}"
fi
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}DRY RUN: Jobs will not be submitted${NC}"
fi
echo -e "${BLUE}========================================${NC}"
echo ""

# Run Python parser and capture output (stdout only, stderr prints directly to terminal)
PARSER_OUTPUT=$(python3 -c "$PARSER_SCRIPT" "$CONFIG_FILE" "$FILTER" "$DEBUG_MODE" "$DRY_RUN")

# Check if Python script succeeded
PYTHON_EXIT_CODE=$?
if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo -e "${RED}Error parsing configuration file!${NC}"
    exit 1
fi

# Process each experiment
EXPERIMENT_COUNT=0
while IFS= read -r line; do
    if [ "$line" = "---EXPERIMENT_END---" ]; then
        # Create and submit job for this experiment
        EXPERIMENT_COUNT=$((EXPERIMENT_COUNT + 1))
        
        JOB_SCRIPT="${JOB_DIR}/run_${EXP_NAME}.sh"
        
        echo -e "${YELLOW}[$EXPERIMENT_COUNT] Creating job: ${EXP_NAME}${NC}"
        echo -e "    Description: ${DESCRIPTION}"
        echo -e "    Nodes: ${NODES}, GPUs/node: ${GPUS_PER_NODE}, Time: ${TIME}"
        echo -e "    Model: hidden_dim=${HIDDEN_DIM}, layers=${MESH_GNN_LAYERS}"
        echo -e "    Training: batch=${BATCH_SIZE}, lr=${LR}, epochs=${MAX_EPOCHS}"
        
        # Build training command arguments
        TRAIN_ARGS="--batch_size ${BATCH_SIZE} \\
    --devices ${GPUS_PER_NODE} \\
    --accelerator gpu \\
    --num_nodes ${NODES} \\
    --hidden_dim ${HIDDEN_DIM} \\
    --mesh_gnn_layers ${MESH_GNN_LAYERS} \\
    --lr ${LR} \\
    --max_epochs ${MAX_EPOCHS} \\
    --exp_name ${EXP_NAME}"
        
        # Add optional parameters
        [ -n "$STRATEGY" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --strategy ${STRATEGY}"
        [ -n "$PRECISION" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --precision ${PRECISION}"
        [ -n "$NUM_WORKERS" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --num_workers ${NUM_WORKERS}"
        [ -n "$NUM_HEADS" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --num_heads ${NUM_HEADS}"
        [ -n "$NUM_PROCESSOR_ROLLOUT_STEPS" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --num_processor_rollout_steps ${NUM_PROCESSOR_ROLLOUT_STEPS}"
        [ "$USE_GAT_ENCODER_DECODER" = "true" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --use_gat_encoder_decoder"
        [ -n "$GAT_CHUNK_SIZE" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --gat_chunk_size ${GAT_CHUNK_SIZE}"
        [ -n "$HIDDEN_LAYERS" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --hidden_layers ${HIDDEN_LAYERS}"
        [ -n "$VALIDATION_INTERVAL" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --validation_interval ${VALIDATION_INTERVAL}"
        [ -n "$LIMIT_VAL_BATCHES" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --limit_val_batches ${LIMIT_VAL_BATCHES}"
        [ -n "$WEIGHT_DECAY" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --weight_decay ${WEIGHT_DECAY}"
        [ -n "$LOSS" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --loss ${LOSS}"
        [ -n "$DATA_PATH" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --data_path ${DATA_PATH}"
        [ -n "$START_DATE" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --start_date ${START_DATE}"
        [ -n "$END_DATE" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --end_date ${END_DATE}"
        [ -n "$DELTA_TIME" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --delta_time ${DELTA_TIME}"
        [ -n "$VERSION" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --version ${VERSION}"
        [ "$MONITOR_MEMORY" = "true" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --monitor_memory"
        [ -n "$MEMORY_LOG_INTERVAL" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --memory_log_interval ${MEMORY_LOG_INTERVAL}"
        [ "$PROFILE" = "true" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --profile"
        [ "$DEBUG_GRADIENTS" = "true" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --debug_gradients"
        [ -n "$NUM_TARGET_BINS" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --num_target_bins ${NUM_TARGET_BINS}"
        [ "$NO_OBS_AUTOREGRESSIVE" = "true" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --no_obs_autoregressive"
        [ "$NO_OBS_AR_TEACHER_FORCING" = "true" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --no_obs_ar_teacher_forcing"
        [ "$NORMALIZE_INCREMENT" = "true" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --normalize_increment"
        [ "$TARGET_IS_ANAL" = "true" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --target_is_anal"
        [ -n "$GRADIENT_LOSS_WEIGHT" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --gradient_loss_weight ${GRADIENT_LOSS_WEIGHT}"
        [ -n "$MULTISCALE_LOSS_WEIGHT" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --multiscale_loss_weight ${MULTISCALE_LOSS_WEIGHT}"
        [ -n "$MULTISCALE_SCALES" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --multiscale_scales ${MULTISCALE_SCALES}"
        [ -n "$ACTION" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --action ${ACTION}"
        [ -n "$LOAD_CKPT_PATH" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --load_ckpt_path ${LOAD_CKPT_PATH}"
        [ -n "$PREDICT_OUTPUT" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --predict_output ${PREDICT_OUTPUT}"
        [ "$NO_EARLY_STOPPING" = "true" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --no_early_stopping"
        [ -n "$EARLY_STOPPING_PATIENCE" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --early_stopping_patience ${EARLY_STOPPING_PATIENCE}"
        [ "$HIERARCHICAL" = "true" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --hierarchical"
        [ -n "$HIERARCHICAL_PROCESSOR_TYPE" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --hierarchical_processor_type ${HIERARCHICAL_PROCESSOR_TYPE}"
        [ -n "$MESH_SPLITS" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --mesh_splits ${MESH_SPLITS}"
        [ -n "$LEVELS" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --levels ${LEVELS}"
        [ -n "$PROCESSOR_WINDOW" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --processor_window ${PROCESSOR_WINDOW}"
        [ -n "$PROCESSOR_DEPTH" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --processor_depth ${PROCESSOR_DEPTH}"
        [ -n "$PROCESSOR_HEADS" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --processor_heads ${PROCESSOR_HEADS}"
        [ -n "$PROCESSOR_DROPOUT" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --processor_dropout ${PROCESSOR_DROPOUT}"
        [ -n "$SPATIAL_MIXING_STEPS" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --spatial_mixing_steps ${SPATIAL_MIXING_STEPS}"
        [ -n "$PROCESSOR_ATTN_CHUNK_SIZE" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --processor_attn_chunk_size ${PROCESSOR_ATTN_CHUNK_SIZE}"
        [ "$NO_PROCESSOR_CAUSAL_MASK" = "true" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --no_processor_causal_mask"
        [ "$NO_PROCESSOR_CROSS_SCALE" = "true" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --no_processor_cross_scale"
        [ -n "$OBS_CONFIG" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --obs_config ${OBS_CONFIG}"
        [ -n "$EXP_TYPE" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --exp_type ${EXP_TYPE}"
        [ -n "$MODEL_PATH" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --model_path ${MODEL_PATH}"
        [ -n "$X_MIN" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --x_min ${X_MIN}"
        [ -n "$X_MAX" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --x_max ${X_MAX}"
        [ -n "$Y_MIN" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --y_min ${Y_MIN}"
        [ -n "$Y_MAX" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --y_max ${Y_MAX}"
        [ -n "$MESH_FEATURE_DIM" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --mesh_feature_dim ${MESH_FEATURE_DIM}"
        [ -n "$CUTOFF_FACTOR" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --cutoff_factor ${CUTOFF_FACTOR}"
        [ -n "$NUM_NEIGHBORS" ] && TRAIN_ARGS="$TRAIN_ARGS \\
    --num_neighbors ${NUM_NEIGHBORS}"

        # Create job script
        cat > "$JOB_SCRIPT" << EOF
#!/bin/bash -l

##SBATCH -A gpu-ai4wp
#SBATCH -A ${ACCOUNT}
#SBATCH -p ${PARTITION}
#SBATCH -q gpu
#SBATCH --gres=gpu:h100:${GPUS_PER_NODE}
#SBATCH -J olt_${EXP_NAME}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=${GPUS_PER_NODE}
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH -t ${TIME}
#SBATCH -o jobs/train_${EXP_NAME}_%J.out
#SBATCH -e jobs/train_${EXP_NAME}_%J.err
#SBATCH --exclude=u22g08,u22g09,u22g10,u23g01

# === Load Environment ===
source /home/Xin.C.Jin/modules/env_diatom.sh

# === Diagnostics ===
echo "=========================================="
echo "Experiment: ${EXP_NAME}"
echo "Description: ${DESCRIPTION}"
echo "Nodes: ${NODES}"
echo "GPUs per node: ${GPUS_PER_NODE}"
echo "Time Limit: ${TIME}"
echo "Hidden Dim: ${HIDDEN_DIM}"
echo "Mesh GNN Layers: ${MESH_GNN_LAYERS}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Learning Rate: ${LR}"
echo "Max Epochs: ${MAX_EPOCHS}"
echo "Hierarchical: ${HIERARCHICAL:-false}"
if [ "${HIERARCHICAL:-false}" = "true" ]; then
  echo "Hierarchical processor type: ${HIERARCHICAL_PROCESSOR_TYPE:-}"
  echo "mesh_splits=${MESH_SPLITS:-} levels=${LEVELS:-}"
fi
echo "=========================================="
echo "Running on \$(hostname)"
echo "SLURM Node List: \$SLURM_NODELIST"
echo "Job ID: \$SLURM_JOB_ID"
nvidia-smi
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

# Environment variables
export OMP_NUM_THREADS=1
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONFAULTHANDLER=1
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_BLOCKING_WAIT=1
export OCELOT_LOG_LEVEL=DEBUG

# === Run Training (DDP) ===
echo "=========================================="
echo "Executing training command:"
echo "=========================================="

CMD="srun --cpu_bind=cores \\
    python -m ocelot.train_hetero_model \\
    ${TRAIN_ARGS}"

echo "\$CMD"
echo "=========================================="
echo ""

# Execute the command
eval \$CMD

echo "=========================================="
echo "Training completed!"
echo "=========================================="
EOF
        
        chmod +x "$JOB_SCRIPT"
        echo -e "    ${GREEN}✓${NC} Job script created: $JOB_SCRIPT"
        
        # Submit job (unless dry-run)
        if [ "$DRY_RUN" = false ]; then
            # Check if sbatch is available
            if ! command -v sbatch &> /dev/null; then
                echo -e "    ${RED}✗${NC} Failed to submit job"
                echo -e "    ${RED}Error:${NC} sbatch command not found. Are you on an HPC login node?"
                echo -e "    ${YELLOW}Tip:${NC} Use --dry-run to test without submitting"
            else
                SUBMIT_OUTPUT=$(sbatch "$JOB_SCRIPT" 2>&1)
                SUBMIT_EXIT_CODE=$?
                
                if [ $SUBMIT_EXIT_CODE -eq 0 ]; then
                    # Extract job ID from output like "Submitted batch job 12345"
                    JOB_ID=$(echo "$SUBMIT_OUTPUT" | grep -oE '[0-9]+' | tail -1)
                    if [ -n "$JOB_ID" ]; then
                        echo -e "    ${GREEN}✓${NC} Job submitted with ID: ${GREEN}$JOB_ID${NC}"
                    else
                        echo -e "    ${GREEN}✓${NC} Job submitted: $SUBMIT_OUTPUT"
                    fi
                else
                    echo -e "    ${RED}✗${NC} Failed to submit job"
                    echo -e "    ${RED}Error:${NC} $SUBMIT_OUTPUT"
                fi
            fi
        else
            echo -e "    ${YELLOW}[DRY RUN]${NC} Would submit: sbatch $JOB_SCRIPT"
        fi
        echo ""
        
    else
        # Parse variable assignment
        eval "$line"
    fi
done <<< "$PARSER_OUTPUT"

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Processed $EXPERIMENT_COUNT experiment(s)${NC}"
echo -e "${BLUE}========================================${NC}"
