from dataset import ModelNetDataLoader, PAPNetDataLoader
from torch.cuda.amp import GradScaler
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import importlib
import shutil
import hydra
import omegaconf

scaler = GradScaler()
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        pred = classifier(points)
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = (
                pred_choice[target == cat]
                .eq(target[target == cat].long().data)
                .cpu()
                .sum()
            )
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


@hydra.main(config_path='config', config_name='cls')
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)

    # HYPER PARAMETERS
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)

    print(omegaconf.OmegaConf.to_yaml(args))

    # DATA LOADING
    logger.info('Load dataset ...')
    DATA_PATH = hydra.utils.to_absolute_path(args.data_path)
    DatasetClass = (
        PAPNetDataLoader if args.use_papnet_loader else ModelNetDataLoader
    )

    TEST_DATASET = DatasetClass(
        root=DATA_PATH,
        npoint=args.num_point,
        split='test',
        normal_channel=args.normal,
    )
    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
        pin_memory=True
    )

    # MODEL LOADING
    args.num_class = 40
    args.input_dim = 6 if args.normal else 3
    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')

    classifier = getattr(
        importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerCls'
    )(args).cuda()

    checkpoint = torch.load(args.checkpoint_path)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Eval
    with torch.no_grad():
        instance_acc, class_acc = test(classifier, testDataLoader)

    return (instance_acc, class_acc)


if __name__ == '__main__':
    main()
