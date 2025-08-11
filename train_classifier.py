from shutil import rmtree
import os
# Reduce VRAM usage by reducing fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
#os.environ["TORCHDYNAMO_VERBOSE"] = "1"


from clearml import Task
from pytorch_lightning.cli import LightningCLI
from models.cnn_classifier import CNNClassifier, ConvFormerClassifier
from models.poolformer_classifier import PoolFormerClassifier, FlexFormerClassifier, MetaFormerClassifier
from models.pretrained_metaformer_classifier import PretrainedMetaformer
from models.metaformer_classifier import AdaptiveMetaformerClassifier
from datasets.med_mnist_dataset import MedMNISTDataModule
from datasets.imagewoof_dataset import ImageWoofDataModule

task = Task.init(project_name="FlexConv/Classification", auto_resource_monitoring=False, reuse_last_task_id=False,
                 auto_connect_frameworks=False)

# training routine
cli = LightningCLI()

# housekeeping
trainer = cli.trainer
Task.current_task().upload_artifact("best.ckpt", trainer.checkpoint_callback.best_model_path, wait_on_upload=True)
Task.current_task().close()
if trainer.logger is not None:
    rmtree(trainer.logger.log_dir)
