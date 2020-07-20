from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
# from utils.prune_utils import *
from test import evaluate

# 调试用的模块，reload用于代码热重载
from importlib import reload
# import debug_utils

from terminaltables import AsciiTable
import warnings
warnings.filterwarnings("ignore")

import os
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3-mobilenet2-dota.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/dota.data", help="path to data config file")
    # parser.add_argument("--pretrained_weights", type=str, default="weights/darknet53.conv.74",
    parser.add_argument("--pretrained_weights", '-pre', type=str,default="weights/yolov3_ckpt.pth", help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=10, help="interval evaluations on validation set")
    parser.add_argument("--multiscale_training", default=False, help="allow for multi-scale training")
    parser.add_argument("--debug_file", type=str, default="debug", help="enter ipdb if dir exists")
    parser.add_argument('--learning_rate', '-lr', dest='lr', type=float, default=1e-3, help='initial learning rate')


    opt = parser.parse_args()
    print(opt)

    logger = Logger("logs")

    # 设置随机数种子
    init_seeds()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%m%d%H%M')
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Model(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))


    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.95)

    metrics = [
        "grid_size",
        "loss",
        "x", "y", "w", "h",
        "conf",
        "cls", "cls_acc",
        "recall50", "recall75",
        "precision",
        "conf_obj", "conf_noobj",
    ]

    for epoch in range(opt.epochs):

        # 进入调试模式
        if os.path.exists(opt.debug_file):
            import ipdb

            ipdb.set_trace()

        model.train()
        start_time = time.time()


        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = imgs.to(device)
            targets = targets.to(device)

            loss, outputs = model(imgs, targets)

            optimizer.zero_grad()
            loss.backward()


            optimizer.step()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            formats = {m: "%.6f" for m in metrics}
            formats["grid_size"] = "%2d"
            formats["cls_acc"] = "%.2f%%"
            for metric in metrics:
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            if batch_i % 50 == 0:
                print(log_str)

            # Tensorboard logging
            tensorboard_log = []
            for i, yolo in enumerate(model.yolo_layers):
                for name, metric in yolo.metrics.items():
                    # 选择部分指标写入tensorboard
                    if name not in {"grid_size", "x", "y", "w", "h", "cls_acc"}:
                        tensorboard_log += [(f"{name}_{i + 1}", metric)]
            tensorboard_log += [("loss", loss.item())]
            tensorboard_log += [("lr", optimizer.param_groups[0]['lr'])]
            logger.list_of_scalars_summary('train', tensorboard_log, batches_done)

        if epoch % opt.evaluation_interval == 0 :
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.01,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=4,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary('valid', evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")
            #print(f"---- mAP {AP.mean()}")


        if epoch % opt.checkpoint_interval == 0 or epoch == opt.epochs - 1:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_{epoch}_{timestamp}.pth")