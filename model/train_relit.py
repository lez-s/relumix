import os
os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
import sys
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from main import SetupCallback, instantiate_from_config, get_parser, WandbLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
import datetime
from pytorch_lightning.utilities import rank_zero_only
from PIL import Image
import torchvision
import numpy as np
from os.path import exists
class RelitImageLogger(pl.Callback):
    def __init__(
        self,
        batch_frequency,
        max_images,
        clamp=True,
        increase_log_steps=True,
        rescale=True,
        disabled=False,
        log_on_batch_idx=False,
        log_first_step=True,
        log_images_kwargs=None,
        log_before_first_step=True,
        enable_autocast=True,
        log_train=True,
    ):
        super().__init__()
        self.enable_autocast = enable_autocast
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.log_steps = [2**n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.log_before_first_step = log_before_first_step
        self.log_train = log_train

    @rank_zero_only
    def log_local(
        self,
        save_dir,
        split,
        images,
        global_step,
        current_epoch,
        batch_idx,
        pl_module=None,
    ):
        root = os.path.join(save_dir, "images", split)
        os.makedirs(root, exist_ok=True)
        
        def make_grid_image(images, nrow=4):
            grid = torchvision.utils.make_grid(images, nrow=nrow)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.cpu().numpy()
            grid = (grid * 255).astype(np.uint8)
            return grid
        
        def process_tensor(name, tensor):
            if isinstance(tensor, torch.Tensor) and len(tensor.shape) >= 4:
                B, *rest = tensor.shape
                if len(rest) == 3:  
                    grid = make_grid_image(tensor)
                elif len(rest) == 4:  
                    B, T = tensor.shape[:2]
                    grid = make_grid_image(tensor.reshape(B * T, *tensor.shape[2:]))
                else:
                    return
                    
                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                    name, global_step, current_epoch, batch_idx
                )
                path = os.path.join(root, filename)
                Image.fromarray(grid).save(path)

                if pl_module is not None: 
                    pl_module.logger.log_image(
                        key=f"{split}/{name}",
                        images=[Image.fromarray(grid)],
                        step=pl_module.global_step,
                    )
        
        def process_dict(prefix, d):
            for k, v in d.items():
                full_key = f"{prefix}_{k}" if prefix else k
                if isinstance(v, dict):
                    process_dict(full_key, v)
                elif isinstance(v, torch.Tensor):
                    process_tensor(full_key, v)
        
        process_dict("", images)

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
            check_idx > 0 or self.log_first_step
        ):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    @rank_zero_only 
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (
            self.check_frequency(check_idx)
            and hasattr(pl_module, "log_images")
            and callable(pl_module.log_images)
            and self.max_images > 0
        ):
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            gpu_autocast_kwargs = {
                "enabled": self.enable_autocast,
                "dtype": torch.get_autocast_gpu_dtype(),
                "cache_enabled": torch.is_autocast_cache_enabled(),
            }

            with torch.no_grad(), torch.cuda.amp.autocast(**gpu_autocast_kwargs):
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                if isinstance(images[k], torch.Tensor):
                    N = min(images[k].shape[0], self.max_images)  
                    images[k] = images[k][:N]
                    images[k] = images[k].detach().float().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1.0, 1.0)

            self.log_local(
                pl_module.logger.save_dir,
                split,
                images,
                pl_module.global_step,
                pl_module.current_epoch,
                batch_idx,
                pl_module=pl_module if isinstance(pl_module.logger, WandbLogger) else None,
            )

            if is_train:
                pl_module.train()

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled and self.log_train:
            self.log_img(pl_module, batch, batch_idx, split="train")

    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")

class CheckpointPathCallback(pl.callbacks.Callback):
    def __init__(self, checkpoint_log_path):
        super().__init__()
        self.checkpoint_log_path = checkpoint_log_path

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if trainer.global_rank == 0:
            # 获取当前步数命名的checkpoint路径
            step_checkpoint_path = os.path.join(
                trainer.checkpoint_callback.dirpath, 
                f"step-{pl_module.global_step}.ckpt"
            )
            # 如果文件存在则记录下来
            if os.path.exists(step_checkpoint_path):
                with open(self.checkpoint_log_path, "a") as f:
                    f.write('\n')
                    f.write(step_checkpoint_path)
            # 同时记录last.ckpt
            last_checkpoint_path = os.path.join(trainer.checkpoint_callback.dirpath, "last.ckpt")
            if os.path.exists(last_checkpoint_path):
                with open(self.checkpoint_log_path, "a") as f:
                    f.write('\n')
                    f.write(last_checkpoint_path)

def main():
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    seed_everything(opt.seed)
    if not opt.base:
        opt.base = ["configs/train_kubric_relit.yaml"]
    
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    if opt.name:
        name = "_" + opt.name
    elif opt.base:
        cfg_fname = os.path.split(opt.base[0])[-1]
        cfg_name = os.path.splitext(cfg_fname)[0]
        name = "_" + cfg_name
    else:
        name = ""
        
    nowname = now + name + opt.postfix
    logdir = os.path.join(opt.logdir, nowname)
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    
    # Add dataset splits directory
    split_save_dir = os.path.join(logdir, "dataset_splits")

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())


    if opt.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = instantiate_from_config(config.model)

    default_logger_cfg = {
        "wandb": {
            "target": "pytorch_lightning.loggers.WandbLogger",
            "params": {
                "name": nowname,
                "offline": opt.debug,
                "id": nowname,
                "project": opt.projectname,
                "log_model": False,
            },
        },
        "csv": {
            "target": "pytorch_lightning.loggers.CSVLogger",
            "params": {
                "name": "testtube",
                "save_dir": logdir,
            },
        },
    }
    logger = instantiate_from_config(
        default_logger_cfg["wandb" if opt.wandb else "csv"]
    )

    callbacks = [
        SetupCallback(
            resume=opt.resume,
            now=now,
            logdir=logdir,
            ckptdir=ckptdir,
            cfgdir=cfgdir,
            config=config,
            lightning_config=lightning_config,
            debug=opt.debug,
        ),
        RelitImageLogger(
            batch_frequency=lightning_config.callbacks.image_logger.params.batch_frequency,
            max_images=lightning_config.callbacks.image_logger.params.max_images,
            clamp=True,
            log_images_kwargs={"include_light_info": True},
        ),
        
        pl.callbacks.ModelCheckpoint(
            dirpath=ckptdir,
            filename="{step}",
            save_last=lightning_config.modelcheckpoint.params.save_last,
            every_n_train_steps=lightning_config.modelcheckpoint.params.every_n_train_steps,
            save_top_k=lightning_config.modelcheckpoint.params.save_top_k,
        ),
        CheckpointPathCallback(
            checkpoint_log_path=lightning_config.callbacks.checkpoint_path_logger.path
        ),
    ]


    trainer_config = {
        "accelerator": "gpu",
        "devices": opt.devices if hasattr(opt, "devices") else 1,
        "precision": 16,
        "max_epochs": lightning_config.trainer.max_epochs,
        "accumulate_grad_batches": lightning_config.trainer.accumulate_grad_batches,
        "logger": logger,
        "callbacks": callbacks,
        "num_sanity_val_steps": lightning_config.trainer.num_sanity_val_steps,
        "benchmark": lightning_config.trainer.benchmark,
    }

 
    bs = config.data.params.batch_size
    base_lr = config.model.base_learning_rate
    ngpu = len(str(trainer_config["devices"]).strip(",").split(","))
    if opt.scale_lr:
        model.learning_rate = trainer_config["accumulate_grad_batches"] * ngpu * bs * base_lr
    else:
        model.learning_rate = base_lr


    data = instantiate_from_config(config.data)
    
    # Add split_save_dir to data module if it supports it
    if hasattr(data, 'split_save_dir'):
        data.split_save_dir = split_save_dir
    elif hasattr(data, '__dict__'):
        # Try to add it directly if the object allows
        try:
            data.split_save_dir = split_save_dir
        except:
            print(f"[yellow]Warning: Could not set split_save_dir on data module")
    
    trainer = Trainer(**trainer_config)


    trainer.fit(model, data, ckpt_path=opt.resume_from_checkpoint)


if __name__ == "__main__":
    main()
