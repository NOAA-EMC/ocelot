import os
import csv
import torch
import time
from pytorch_lightning import Callback


class CombinedMemoryCallback(Callback):
    """
    Comprehensive memory monitoring and debugging callback.

    Combines features from both MemoryMonitorCallback and FullMemoryDebugCallback:
    - Detailed per-GPU memory snapshots
    - Step-by-step memory leak detection
    - Per-module memory tracking (encoder/processor/decoder)
    - Gradient norm tracking
    - Active tensor allocation counting
    - Epoch-level summaries
    - CSV logging (optional)
    - Memory fragmentation analysis
    """

    def __init__(self, log_every_n_steps=10, detailed=False, csv_path=None):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.detailed = detailed
        self.csv_path = csv_path
        self.prev_alloc = None

        # Initialize CSV file if path provided
        if csv_path:
            # Create parent directory if it doesn't exist
            csv_dir = os.path.dirname(csv_path)
            if csv_dir:
                os.makedirs(csv_dir, exist_ok=True)

            # Create CSV file with header if it doesn't exist
            if not os.path.exists(csv_path):
                with open(csv_path, "w") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "step",
                        "epoch",
                        "alloc_GB",
                        "reserved_GB",
                        "increase_MB",
                        "fragmentation_GB",
                        "grad_norm",
                        "encoder_MB",
                        "processor_MB",
                        "decoder_MB",
                        "num_allocations",
                        "timestamp"
                    ])

    # -----------------------
    # HELPERS
    # -----------------------

    def _rank0(self, trainer):
        """Return True only on global rank zero."""
        return trainer.global_rank == 0

    def _get_module_memory(self, model):
        """
        Returns PARAMETER memory only (static, won't change during training).
        Does NOT include activations, gradients, or optimizer states.
        Useful for verifying model architecture, not for detecting memory leaks.
        """

        def size_mb(module):
            return sum(p.numel() * p.element_size()
                       for p in module.parameters()) / (1024 ** 2)

        return {
            "encoder": size_mb(
                model.observation_encoders) if hasattr(
                model,
                "observation_encoders") else 0,
            "processor": size_mb(
                model.processor) if hasattr(
                    model,
                    "processor") else 0,
            "decoder": size_mb(
                model.observation_decoders) if hasattr(
                model,
                "observation_decoders") else 0,
        }

    def _tensor_snapshot_growth(self):
        """Track number of active CUDA memory allocations."""
        try:
            if not torch.cuda.is_available():
                return 0
            stats = torch.cuda.memory_stats()
            # Get current number of active allocations
            num_allocs = stats.get('active.all.current', 0)
            return num_allocs
        except Exception as e:
            print(f"Warning: Could not get tensor snapshot: {e}")
            return 0

    def _grad_norm(self, model):
        """Calculate L2 norm of all gradients."""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def _print_memory_summary(self, stage_name):
        """
        Print detailed GPU memory usage for all GPUs.
        From MemoryMonitorCallback's print_memory_summary function.
        """
        if not torch.cuda.is_available():
            return

        num_gpus = torch.cuda.device_count()

        print("\n" + "=" * 80)
        print(f"🔍 MEMORY SNAPSHOT: {stage_name}")
        print("=" * 80)

        for gpu_id in range(num_gpus):
            torch.cuda.set_device(gpu_id)

            # Get memory stats
            allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)    # GB
            max_allocated = torch.cuda.max_memory_allocated(
                gpu_id) / (1024**3)  # GB
            total = torch.cuda.get_device_properties(
                gpu_id).total_memory / (1024**3)  # GB
            free = total - allocated

            print(f"\n📊 GPU {gpu_id}:")
            print(f"   Total Memory:     {total:.2f} GB")
            print(
                f"   Allocated:        {allocated:.2f} GB ({allocated/total*100:.1f}%)")
            print(
                f"   Reserved:         {reserved:.2f} GB ({reserved/total*100:.1f}%)")
            print(
                f"   Free:             {free:.2f} GB ({free/total*100:.1f}%)")
            print(
                f"   Peak Allocated:   {max_allocated:.2f} GB ({max_allocated/total*100:.1f}%)")
            print(f"   Fragmentation:    {(reserved - allocated):.2f} GB")

            # Get detailed memory breakdown if available
            try:
                memory_stats = torch.cuda.memory_stats(gpu_id)
                active_bytes = memory_stats.get(
                    'active_bytes.all.current', 0) / (1024**3)
                inactive_bytes = memory_stats.get(
                    'inactive_split_bytes.all.current', 0) / (1024**3)

                print(f"   Active Tensors:   {active_bytes:.2f} GB")
                print(f"   Inactive/Cached:  {inactive_bytes:.2f} GB")
            except BaseException:
                pass

        print("=" * 80 + "\n")

    # -----------------------
    # HOOKS
    # -----------------------

    def on_train_epoch_start(self, trainer, pl_module):
        """Log memory at the start of each epoch."""
        if self._rank0(trainer):
            if self.detailed:
                self._print_memory_summary(
                    f"Epoch {trainer.current_epoch} Start")
            else:
                if torch.cuda.is_available():
                    alloc = torch.cuda.memory_allocated(0) / (1024**3)
                    reserved = torch.cuda.memory_reserved(0) / (1024**3)
                    print(
                        f"\n🔹 Epoch {trainer.current_epoch} Start - Memory: {alloc:.2f} GB allocated, {reserved:.2f} GB reserved")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Log memory before forward pass (detailed mode only)."""
        if not self._rank0(trainer):
            return

        if batch_idx % self.log_every_n_steps == 0 and self.detailed:
            self._print_memory_summary(
                f"Step {trainer.global_step} - Before Forward")

    def on_train_batch_end(
            self,
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx):
        """Log memory after backward pass with detailed debugging info."""
        if not self._rank0(trainer):
            return

        if batch_idx % self.log_every_n_steps != 0:
            return

        # Check CUDA availability
        if not torch.cuda.is_available():
            print("⚠️ CUDA not available, skipping memory debug")
            return

        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        alloc_mb = alloc * 1024

        # Memory growth
        if self.prev_alloc is None:
            increase_mb = 0.0
        else:
            increase_mb = alloc_mb - self.prev_alloc

        self.prev_alloc = alloc_mb

        # Per-module memory
        module_mem = self._get_module_memory(pl_module)
        enc_mb = module_mem["encoder"]
        proc_mb = module_mem["processor"]
        dec_mb = module_mem["decoder"]

        # Tensor snapshot growth
        num_tensors = self._tensor_snapshot_growth()

        # Gradient norm
        grad_norm = self._grad_norm(pl_module)

        # CUDA summary
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        fragmentation = reserved - alloc

        # ---------------------------------------------
        # PRINT REPORT
        # ---------------------------------------------
        if self.detailed:
            # Detailed snapshot
            self._print_memory_summary(
                f"Step {trainer.global_step} - After Backward")
        else:
            # Quick summary
            print("\n================ GPU MEMORY DEBUG ================")
            print(
                f"Step: {trainer.global_step} | Epoch: {trainer.current_epoch}")
            print(f"Allocated: {alloc:.2f} GB   Reserved: {reserved:.2f} GB")
            print(f"Fragmentation: {fragmentation:.2f} GB")
            print(f"Increase per step: {increase_mb:.1f} MB")
            print(f"Active allocations: {num_tensors}")
            print(f"Gradient L2 norm: {grad_norm:.2e}")
            print("--- Per-module params ---")
            print(f"Encoder:   {enc_mb:.1f} MB")
            print(f"Processor: {proc_mb:.1f} MB")
            print(f"Decoder:   {dec_mb:.1f} MB")
            print("=================================================\n")

        # ---------------------------------------------
        # Write CSV
        # ---------------------------------------------
        if self.csv_path:
            with open(self.csv_path, "a") as f:
                writer = csv.writer(f)
                writer.writerow([
                    trainer.global_step,
                    trainer.current_epoch,
                    alloc,
                    reserved,
                    increase_mb,
                    fragmentation,
                    grad_norm,
                    enc_mb,
                    proc_mb,
                    dec_mb,
                    num_tensors,
                    time.time(),
                ])

    def on_train_epoch_end(self, trainer, pl_module):
        """Log memory at the end of each epoch and reset peak stats."""
        if not self._rank0(trainer):
            return

        if self.detailed:
            self._print_memory_summary(f"Epoch {trainer.current_epoch} End")
        else:
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated(0) / (1024**3)
                reserved = torch.cuda.memory_reserved(0) / (1024**3)
                peak = torch.cuda.max_memory_allocated(0) / (1024**3)
                print(
                    f"\n🔹 Epoch {trainer.current_epoch} End - Memory: {alloc:.2f} GB allocated, Peak: {peak:.2f} GB\n")

        # Reset peak memory stats for next epoch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
