

import torch
from solo.methods.mae import MAE

from .utils import gen_base_cfg, gen_batch, gen_trainer, prepare_dummy_dataloaders


def test_mae():
    method_kwargs = {
        "decoder_embed_dim": 512,
        "decoder_depth": 8,
        "decoder_num_heads": 16,
        "mask_ratio": 0.75,
        "norm_pix_loss": True,
    }
    cfg = gen_base_cfg("mae", batch_size=2, num_classes=100, momentum=True)
    cfg.method_kwargs = method_kwargs
    cfg.backbone = {"name": "vit_small", "kwargs": {"img_size": 224, "patch_size": 16}}

    model = MAE(cfg)

    # test arguments
    model.add_and_assert_specific_cfg(cfg)

    # test parameters
    assert model.learnable_params is not None

    # test forward
    batch, _ = gen_batch(cfg.optimizer.batch_size, cfg.data.num_classes, "imagenet100")
    out = model(batch[1][0])
    assert (
        "logits" in out
        and isinstance(out["logits"], torch.Tensor)
        and out["logits"].size() == (cfg.optimizer.batch_size, cfg.data.num_classes)
    )
    assert (
        "feats" in out
        and isinstance(out["feats"], torch.Tensor)
        and out["feats"].size() == (cfg.optimizer.batch_size, model.features_dim)
    )
    assert (
        "pred" in out
        and isinstance(out["pred"], torch.Tensor)
        and out["pred"].size() == (cfg.optimizer.batch_size, 14 * 14, 16 * 16 * 3)
    )
    assert (
        "mask" in out
        and isinstance(out["mask"], torch.Tensor)
        and out["mask"].size() == (cfg.optimizer.batch_size, 14 * 14)
    )

    # imagenet
    model = MAE(cfg)
    trainer = gen_trainer(cfg)
    train_dl, val_dl = prepare_dummy_dataloaders(
        "imagenet100",
        num_large_crops=cfg.data.num_large_crops,
        num_small_crops=0,
        num_classes=cfg.data.num_classes,
        batch_size=cfg.optimizer.batch_size,
    )
    trainer.fit(model, train_dl, val_dl)

    # cifar
    cfg.data.dataset = "COCO"
    cfg.data.num_classes = 80
    cfg.backbone = {"name": "vit_small", "kwargs": {"img_size": 32, "patch_size": 8}}
    model = MAE(cfg)

    trainer = gen_trainer(cfg)
    train_dl, val_dl = prepare_dummy_dataloaders(
        "cifar10",
        num_large_crops=cfg.data.num_large_crops,
        num_small_crops=0,
        num_classes=cfg.data.num_classes,
        batch_size=cfg.optimizer.batch_size,
    )
    trainer.fit(model, train_dl, val_dl)
