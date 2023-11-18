

import torch
from solo.methods import SimCLR

from .utils import gen_base_cfg, gen_batch, gen_trainer, prepare_dummy_dataloaders


def test_simclr():
    method_kwargs = {
        "proj_output_dim": 256,
        "proj_hidden_dim": 2048,
        "temperature": 0.2,
        "supervised": False,
    }

    cfg = gen_base_cfg("simclr", batch_size=2, num_classes=100, momentum=True)
    cfg.method_kwargs = method_kwargs
    model = SimCLR(cfg)

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
        "z" in out
        and isinstance(out["z"], torch.Tensor)
        and out["z"].size() == (cfg.optimizer.batch_size, method_kwargs["proj_output_dim"])
    )

    multicrop_out = model.multicrop_forward(batch[1][0])
    assert (
        "feats" in multicrop_out
        and isinstance(multicrop_out["feats"], torch.Tensor)
        and multicrop_out["feats"].size() == (cfg.optimizer.batch_size, model.features_dim)
    )
    assert (
        "z" in multicrop_out
        and isinstance(multicrop_out["z"], torch.Tensor)
        and multicrop_out["z"].size()
        == (cfg.optimizer.batch_size, method_kwargs["proj_output_dim"])
    )

    # imagenet
    model = SimCLR(cfg)

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
    cfg.data.dataset = "cifar10"
    cfg.data.num_classes = 10
    model = SimCLR(cfg)

    trainer = gen_trainer(cfg)
    train_dl, val_dl = prepare_dummy_dataloaders(
        "cifar10",
        num_large_crops=cfg.data.num_large_crops,
        num_small_crops=0,
        num_classes=cfg.data.num_classes,
        batch_size=cfg.optimizer.batch_size,
    )
    trainer.fit(model, train_dl, val_dl)

    # multicrop
    cfg.data.num_small_crops = 6
    model = SimCLR(cfg)

    trainer = gen_trainer(cfg)
    train_dl, val_dl = prepare_dummy_dataloaders(
        "imagenet100",
        num_large_crops=cfg.data.num_large_crops,
        num_small_crops=cfg.data.num_small_crops,
        num_classes=cfg.data.num_classes,
        batch_size=cfg.optimizer.batch_size,
    )
    trainer.fit(model, train_dl, val_dl)
