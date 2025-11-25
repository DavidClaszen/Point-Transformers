import torch
import importlib
import hydra
import omegaconf

from dataset import ModelNetDataLoader, PAPNetDataLoader
from torch.cuda.amp import autocast
from omegaconf import OmegaConf


def get_predictions(args):
    """
    Load dataset + model from args, and return (targets, preds) as numpy arrays.
    """

    # ---- DATA LOADING ----
    OmegaConf.set_struct(args, False)
    DATA_PATH = hydra.utils.to_absolute_path(args.data_path)

    DatasetClass = PAPNetDataLoader if args.use_papnet_loader else ModelNetDataLoader

    TEST_DATASET = DatasetClass(
        root=DATA_PATH,
        npoint=args.num_point,
        split='test',
        normal_channel=args.normal,
    )

    test_loader = torch.utils.data.DataLoader(
        TEST_DATASET,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    # ---- MODEL LOADING ----
    args.num_class = 40
    args.input_dim = 6 if args.normal else 3

    ModelCls = getattr(
        importlib.import_module(f"models.{args.model.name}.model"),
        "PointTransformerCls",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = ModelCls(args).to(device)

    total_params = sum(p.numel() for p in classifier.parameters())
    print(f"Total number of parameters: {total_params}")

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    classifier.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model loaded with {checkpoint['epoch']} epochs")
    classifier.eval()

    # ---- COLLECT PREDS & TARGETS ----
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for points, target in test_loader:
            target = target[:, 0]
            points = points.to(device)
            target = target.to(device)
            with autocast():
                logits = classifier(points)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(target.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    return all_targets, all_preds


@hydra.main(config_path='config', config_name='cls', version_base='1.2')
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)
    y_true, y_pred = get_predictions(args)
    print("Got", len(y_true), "predictions")


if __name__ == "__main__":
    main()
