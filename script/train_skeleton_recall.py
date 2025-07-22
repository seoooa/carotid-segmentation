import autorootcwd
import lightning.pytorch as pytorch_lightning
from monai.utils import set_determinism
from monai.transforms import (
    Compose,
    EnsureType,
    AsDiscrete,
)
from monai.networks.nets import AttentionUnet, SegResNet, UNETR, SwinUNETR, VNet
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from lightning.pytorch.callbacks import (
    BatchSizeFinder,
    LearningRateFinder,
    StochasticWeightAveraging,
)

from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.config import print_config
import torch
import os
import click
import numpy as np
import nibabel as nib
from pathlib import Path
import csv
from dvclive.lightning import DVCLiveLogger

from src.data.dataloader_skeleton_recall import CarotidSkeletonDataModule
from src.models.networks import NetworkFactory
from src.losses.losses import LossFactory
from src.metrics.metrics import MetricFactory

def print_monai_config():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank == 0:
        print_config()

class CarotidSkeletonModel(pytorch_lightning.LightningModule):
    """Model for skeleton recall loss training"""

    def __init__(
        self,
        arch_name="UNETR",
        loss_fn="DiceSkeletonRecallLoss",
        batch_size=1,
        lr=1e-3,
        patch_size=(96, 96, 96),
        fold_number=1,
    ):
        super().__init__()

        self.fold_number = fold_number

        self._model = NetworkFactory.create_network(arch_name, patch_size)
        self.loss_function = LossFactory.create_loss(loss_fn)
        self.metrics = MetricFactory.create_metrics()

        self.post_pred = Compose(
            [EnsureType("tensor", device="cpu"), AsDiscrete(argmax=True, to_onehot=2)]
        )
        self.post_label = Compose(
            [EnsureType("tensor", device="cpu"), AsDiscrete(to_onehot=2)]
        )
        
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.validation_step_outputs = []
        self.batch_size = batch_size
        self.lr = lr
        self.patch_size = patch_size
        self.result_folder = Path("result")
        self.test_step_outputs = []

    def forward(self, x):
        return self._model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        skeleton = batch.get("skeleton", None)  # Get skeleton if available
        
        output = self.forward(images)
        
        # Handle different loss functions
        loss_fn_name = self.loss_function.__class__.__name__
        
        if loss_fn_name == "SkeletonRecallLossWrapper":
            # For DC_SkelREC_and_CE_loss: forward(input, target, target_skel)
            if skeleton is None:
                raise ValueError("Skeleton data is required for SkeletonRecallLossWrapper")
            loss = self.loss_function(output, labels, target_skel=skeleton)
            
        elif loss_fn_name == "SoftSkeletonRecallLossWrapper":
            # For pure skeleton recall loss: forward(input, target_skel, loss_mask=None)
            if skeleton is None:
                raise ValueError("Skeleton data is required for SoftSkeletonRecallLossWrapper")
            loss = self.loss_function(output, skeleton)
            
        else:
            # For standard losses without skeleton (DiceLoss, DiceCELoss, etc.)
            loss = self.loss_function(output, labels)
        
        metrics = loss.item()
        self.log(
            "train_loss",
            metrics,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        skeleton = batch.get("skeleton", None)
        
        roi_size = self.patch_size
        sw_batch_size = 4
        outputs = sliding_window_inference(
            images, roi_size, sw_batch_size, self.forward
        )
        
        # Handle different loss functions (same logic as training_step)
        loss_fn_name = self.loss_function.__class__.__name__
        
        if loss_fn_name == "SkeletonRecallLossWrapper":
            # For DC_SkelREC_and_CE_loss: forward(input, target, target_skel)
            if skeleton is None:
                raise ValueError("Skeleton data is required for SkeletonRecallLossWrapper")
            loss = self.loss_function(outputs, labels, target_skel=skeleton)
            
        elif loss_fn_name == "SoftSkeletonRecallLossWrapper":
            # For pure skeleton recall loss: forward(input, target_skel, loss_mask=None)
            if skeleton is None:
                raise ValueError("Skeleton data is required for SoftSkeletonRecallLossWrapper")
            loss = self.loss_function(outputs, skeleton)
            
        else:
            # For standard losses without skeleton (DiceLoss, DiceCELoss, etc.)
            loss = self.loss_function(outputs, labels)
        
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        
        MetricFactory.calculate_metrics(self.metrics, outputs, labels)
        
        d = {"val_loss": loss, "val_number": len(outputs)}
        self.validation_step_outputs.append(d)
        return d

    def on_validation_epoch_end(self):
        val_loss, num_items = 0, 0
        for output in self.validation_step_outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        
        metric_results = MetricFactory.aggregate_metrics(self.metrics)
        MetricFactory.reset_metrics(self.metrics)
            
        mean_val_loss = torch.tensor(val_loss / num_items, device=self.device)
        log_dict = {
            "val_dice": torch.tensor(metric_results["dice"], device=self.device),
            "val_hausdorff": torch.tensor(metric_results["hausdorff"], device=self.device),
            "val_iou": torch.tensor(metric_results["iou"], device=self.device),
            "val_precision": torch.tensor(metric_results["precision"], device=self.device),
            "val_recall": torch.tensor(metric_results["recall"], device=self.device),
            "val_cldice": torch.tensor(metric_results["cldice"], device=self.device),
            "val_betti_0": torch.tensor(metric_results["betti_0"], device=self.device),
            "val_betti_1": torch.tensor(metric_results["betti_1"], device=self.device),
            "val_loss": mean_val_loss,
        }

        self.log_dict(log_dict, sync_dist=True)

        if metric_results["dice"] > self.best_val_dice:
            self.best_val_dice = metric_results["dice"]
            self.best_val_epoch = self.current_epoch
        print(
            f"current epoch: {self.current_epoch} "
            f"current mean dice: {metric_results['dice']:.4f}, "
            f"hausdorff: {metric_results['hausdorff']:.4f}, "
            f"iou: {metric_results['iou']:.4f}, "
            f"precision: {metric_results['precision']:.4f}, "
            f"recall: {metric_results['recall']:.4f}, "
            f"cldice: {metric_results['cldice']:.4f}, "
            f"betti_0: {metric_results['betti_0']:.4f}, "
            f"betti_1: {metric_results['betti_1']:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        roi_size = self.patch_size
        sw_batch_size = 4
        outputs = sliding_window_inference(
            images, roi_size, sw_batch_size, self.forward
        )
       
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        
        filename = batch["image"].meta["filename_or_obj"][0]
        patient_id = filename.split("\\")[-2]
        
        # Calculate metrics
        MetricFactory.calculate_metrics(self.metrics, outputs, labels)
        metric_results = MetricFactory.aggregate_metrics(self.metrics)
        
        d = {
            "test_dice": metric_results["dice"],
            "test_hausdorff": metric_results["hausdorff"],
            "test_iou": metric_results["iou"],
            "test_precision": metric_results["precision"],
            "test_recall": metric_results["recall"],
            "test_cldice": metric_results["cldice"],
            "test_betti_0": metric_results["betti_0"],
            "test_betti_1": metric_results["betti_1"],
            "patient_id": patient_id,
        }
        self.test_step_outputs.append(d)

        MetricFactory.reset_metrics(self.metrics)
        return d

    def on_test_epoch_end(self):
        # Calculate mean metrics
        dice_scores = [x["test_dice"] for x in self.test_step_outputs]
        hausdorff_scores = [x["test_hausdorff"] for x in self.test_step_outputs]
        iou_scores = [x["test_iou"] for x in self.test_step_outputs]
        precision_scores = [x["test_precision"] for x in self.test_step_outputs]
        recall_scores = [x["test_recall"] for x in self.test_step_outputs]
        cldice_scores = [x["test_cldice"] for x in self.test_step_outputs]
        betti_0_scores = [x["test_betti_0"] for x in self.test_step_outputs]
        betti_1_scores = [x["test_betti_1"] for x in self.test_step_outputs]

        # Calculate means
        mean_dice = np.mean(dice_scores)
        mean_hausdorff = np.mean(hausdorff_scores)
        mean_iou = np.mean(iou_scores)
        mean_precision = np.mean(precision_scores)
        mean_recall = np.mean(recall_scores)
        mean_cldice = np.mean(cldice_scores)
        mean_betti_0 = np.mean(betti_0_scores)
        mean_betti_1 = np.mean(betti_1_scores)

        # Log mean metrics
        self.log_dict({
            "test/mean_dice": mean_dice,
            "test/mean_hausdorff": mean_hausdorff,
            "test/mean_iou": mean_iou,
            "test/mean_precision": mean_precision,
            "test/mean_recall": mean_recall,
            "test/mean_cldice": mean_cldice,
            "test/mean_betti_0": mean_betti_0,
            "test/mean_betti_1": mean_betti_1,
        })

        print(f"\nTest Result Summary:")
        print(f"Mean Dice Score: {mean_dice:.4f}")
        print(f"Mean Hausdorff Distance: {mean_hausdorff:.4f}")
        print(f"Mean IoU Score: {mean_iou:.4f}")
        print(f"Mean Precision Score: {mean_precision:.4f}")
        print(f"Mean Recall Score: {mean_recall:.4f}")
        print(f"Mean CLDice Score: {mean_cldice:.4f}")
        print(f"Mean Betti-0 Error: {mean_betti_0:.4f}")
        print(f"Mean Betti-1 Error: {mean_betti_1:.4f}")

        self.test_step_outputs.clear()


@click.command()
@click.option(
    "--arch_name",
    type=click.Choice(
        ["UNet", "AttentionUnet", "SegResNet", "UNETR", "SwinUNETR", "VNet", "DynUNet"]
    ),
    default="SegResNet",
    help="Choose the architecture name for the model.",
)
@click.option(
    "--loss_fn",
    type=click.Choice(["DiceLoss", "DiceCELoss", "DiceFocalLoss", "SkeletonRecallLoss", "DC_SkelREC_and_CE_loss"]),
    default="DC_SkelREC_and_CE_loss",
    help="Choose the loss function for training.",
)
@click.option(
    "--max_epochs",
    type=int,
    default=200,
    help="Set the maximum number of training epochs.",
)
@click.option(
    "--check_val_every_n_epoch",
    type=int,
    default=10,
    help="Set the frequency of validation checks (in epochs).",
)
@click.option(
    "--gpu_number", type=str, default="0", help="GPU number to use (ex. 0 or 0,1,2,3)"
)
@click.option(
    "--checkpoint_path",
    type=str,
    default=None,
    help="Path to a checkpoint file to load for inference.",
)
@click.option(
    "--fold_number", 
    type=int, 
    default=1, 
    help="Specify the fold number for training. (ex. 1, 2, 3, 4, 5)"
)
@click.option(
    "--target",
    type=click.Choice(["carotid", "mandible", "spinalcord", "thyroid"]),
    default="carotid",
    help="Choose the target anatomy for segmentation.",
)
@click.option(
    "--skeleton_tube",
    type=bool,
    default=True,
    help="Whether to apply tube dilation to skeleton",
)
def main(
    arch_name,
    loss_fn,
    max_epochs,
    check_val_every_n_epoch,
    gpu_number,
    checkpoint_path,
    fold_number,
    target,
    skeleton_tube
):
    # NCCL communication
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    
    # multiprocessing
    torch.multiprocessing.set_start_method("spawn", force=True)
    torch.set_float32_matmul_precision('medium')
    
    print_monai_config()

    # set up loggers and checkpoints
    log_dir = f"result/{loss_fn}/{arch_name}/{target}/fold_{fold_number}"
    os.makedirs(log_dir, exist_ok=True)

    # GPU Setting
    if "," in str(gpu_number):
        devices = [int(gpu) for gpu in str(gpu_number).split(",")]
        strategy = "ddp_find_unused_parameters_true"
    else:
        devices = [int(gpu_number)]
        strategy = "auto" 

    # Set up callbacks
    callbacks = [
        StochasticWeightAveraging(
            swa_lrs=[1e-4], annealing_epochs=5, swa_epoch_start=100
        )
    ]
    dvc_logger = DVCLiveLogger(log_model=True, dir=log_dir, report="html")

    # initialise Lightning's trainer.
    trainer = pytorch_lightning.Trainer(
        devices=devices,
        strategy=strategy,
        max_epochs=max_epochs,
        logger=dvc_logger,
        enable_checkpointing=True,
        benchmark=True,
        accumulate_grad_batches=5,
        precision="bf16-mixed",
        check_val_every_n_epoch=check_val_every_n_epoch,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        default_root_dir=log_dir,
    )

    # Initialize data module with skeleton support
    data_module = CarotidSkeletonDataModule(
        data_dir=f"data/Han_Seg_{target.capitalize()}",
        batch_size=1,
        patch_size=(96, 96, 96),
        num_workers=4,
        cache_rate=0.4,
        fold_number=fold_number,
        target=target,
        use_skeleton=True,
        skeleton_do_tube=skeleton_tube
    )
    data_module.prepare_data()

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint_filename = os.path.basename(checkpoint_path)
        
        if "final_model.ckpt" in checkpoint_filename:
            # Test mode
            print("Loading final model for testing...")
            model = CarotidSkeletonModel.load_from_checkpoint(
                checkpoint_path,
                arch_name=arch_name,
                loss_fn=loss_fn,
                batch_size=1,
                fold_number=fold_number
            )
            model.result_folder = Path(log_dir)
            trainer.test(model=model, datamodule=data_module)
        
        elif "epoch=" in checkpoint_filename:
            # Resume training
            print("Resuming training from checkpoint...")
            model = CarotidSkeletonModel(
                arch_name=arch_name,
                loss_fn=loss_fn,
                batch_size=1,
                fold_number=fold_number
            )
            model.result_folder = Path(log_dir)
            trainer.fit(model, datamodule=data_module, ckpt_path=checkpoint_path)
            trainer.save_checkpoint(os.path.join(log_dir, "final_model.ckpt"))
            trainer.test(model=model, datamodule=data_module)
        
        else:
            # Test mode
            print("Unknown checkpoint format, using for testing...")
            model = CarotidSkeletonModel.load_from_checkpoint(
                checkpoint_path,
                arch_name=arch_name,
                loss_fn=loss_fn,
                batch_size=1,
                fold_number=fold_number
            )
            model.result_folder = Path(log_dir)
            trainer.test(model=model, datamodule=data_module)
    else:
        # Initialize model for training
        model = CarotidSkeletonModel(
            arch_name=arch_name,
            loss_fn=loss_fn,
            batch_size=1,
            fold_number=fold_number
        )
        model.result_folder = Path(log_dir)

        # Train the model
        trainer.fit(model, datamodule=data_module)
        trainer.save_checkpoint(os.path.join(log_dir, "final_model.ckpt"))
        trainer.test(model=model, datamodule=data_module)


if __name__ == "__main__":
    main() 