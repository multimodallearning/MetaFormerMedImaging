from shutil import rmtree
import os
# Reduce VRAM usage by reducing fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
#os.environ["TORCHDYNAMO_VERBOSE"] = "1"


from clearml import Task
from pytorch_lightning.cli import LightningCLI
from datasets.grazpedwri_dataset import SegGrazPedWriDataModule
from datasets.jsrt_dataset import JSRTDataModule
from models.poolformer_segmentor import MetaFormerSegmentator
from models.unet import UNetSegmentator

task = Task.init(project_name="FlexConv/Segmentation", auto_resource_monitoring=False, reuse_last_task_id=False,
                 auto_connect_frameworks=False)

# training routine
cli = LightningCLI()

# housekeeping
trainer = cli.trainer
Task.current_task().upload_artifact("best.ckpt", trainer.checkpoint_callback.best_model_path, wait_on_upload=True)
Task.current_task().close()
if trainer.logger is not None:
    rmtree(trainer.logger.log_dir)
