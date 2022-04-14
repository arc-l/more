import numpy as np
import utils
import torch
from models import PushNet, reinforcement_net
from dataset import LifelongDataset
import argparse
import time
import datetime
import cv2
from torchvision.transforms import ToPILImage
import os
from constants import (
    GRIPPER_GRASP_INNER_DISTANCE,
    GRIPPER_GRASP_INNER_DISTANCE_PIXEL,
    GRIPPER_GRASP_OUTER_DISTANCE_PIXEL,
    GRIPPER_GRASP_SAFE_WIDTH_PIXEL,
    GRIPPER_GRASP_WIDTH_PIXEL,
    PUSH_DISTANCE_PIXEL,
    NUM_ROTATION,
)
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

import log_utils
import torch_utils


def parse_args():
    default_params = {
        "lr": 1e-4,
        "batch_size": 8,
        "t_0": 50,  # CosineAnnealing, start 1 21 61
        "t_mult": 2,  # CosineAnnealing, period 20 40
        "eta_min": 1e-8,  # CosineAnnealing, minimum lr
        "epochs": 51,  # CosineAnnealing, should end before warm start
        # "lr": 1e-5,
        # "batch_size": 28,
        # "t_0": 20,  # CosineAnnealing, start 1 21 61
        # "t_mult": 2,  # CosineAnnealing, period 20 40
        # "eta_min": 1e-8,  # CosineAnnealing, minimum lr
        # "epochs": 21,  # CosineAnnealing, should end before warm start
        "loss_beta": 0.8,
        "num_rotation": NUM_ROTATION,
    }

    parser = argparse.ArgumentParser(description="Train lifelong")

    parser.add_argument(
        "--lr",
        action="store",
        type=float,
        default=default_params["lr"],
        help="Enter the learning rate",
    )
    parser.add_argument(
        "--batch_size",
        action="store",
        default=default_params["batch_size"],
        type=int,
        help="Enter the batchsize for training and testing",
    )
    parser.add_argument(
        "--t_0",
        action="store",
        default=default_params["t_0"],
        type=int,
        help="The t_0 of CosineAnnealing",
    )
    parser.add_argument(
        "--t_mult",
        action="store",
        default=default_params["t_mult"],
        type=int,
        help="The t_mult of CosineAnnealing",
    )
    parser.add_argument(
        "--eta_min",
        action="store",
        default=default_params["eta_min"],
        type=float,
        help="The eta_min of CosineAnnealing",
    )
    parser.add_argument(
        "--epochs",
        action="store",
        default=default_params["epochs"],
        type=int,
        help="Enter the epoch for training",
    )
    parser.add_argument(
        "--loss_beta",
        action="store",
        default=default_params["loss_beta"],
        type=int,
        help="The beta of SmoothL1Loss",
    )
    parser.add_argument(
        "--num_rotation",
        action="store",
        default=default_params["num_rotation"],
        type=int,
        help="Number of rotation",
    )
    parser.add_argument("--dataset_root", action="store", help="Enter the path to the dataset")
    parser.add_argument(
        "--pretrained_model", action="store", help="The path to the pretrained model"
    )
    parser.add_argument(
        "--ratio",
        action="store",
        default=1,
        type=float,
        help="ratio of how many data we use",
    )

    args = parser.parse_args()

    return args


def get_prediction_vis(predictions, color_heightmap, best_pix_ind, is_push=False):

    canvas = None
    num_rotations = predictions.shape[0]
    for canvas_row in range(int(num_rotations / 4)):
        tmp_row_canvas = None
        for canvas_col in range(4):
            rotate_idx = canvas_row * 4 + canvas_col
            prediction_vis = predictions[rotate_idx, :, :].copy()
            prediction_vis = np.clip(prediction_vis, 0, 1)
            prediction_vis.shape = (predictions.shape[1], predictions.shape[2])
            prediction_vis = cv2.applyColorMap(
                (prediction_vis * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            if rotate_idx == best_pix_ind[0]:
                prediction_vis = cv2.circle(
                    prediction_vis, (int(best_pix_ind[2]), int(best_pix_ind[1])), 7, (0, 0, 255), 2,
                )
            prediction_vis = utils.rotate(prediction_vis, rotate_idx * (360.0 / num_rotations))
            if rotate_idx == best_pix_ind[0]:
                center = np.array([[[int(best_pix_ind[2]), int(best_pix_ind[1])]]])
                M = cv2.getRotationMatrix2D(
                    (prediction_vis.shape[1] // 2, prediction_vis.shape[0] // 2,),
                    rotate_idx * (360.0 / num_rotations),
                    1,
                )
                center = cv2.transform(center, M)
                center = np.transpose(center[0])
                if is_push:
                    point_from = (int(center[0]), int(center[1]))
                    point_to = (int(center[0] + PUSH_DISTANCE_PIXEL), int(center[1]))
                    prediction_vis = cv2.arrowedLine(
                        prediction_vis, point_from, point_to, (100, 255, 0), 2, tipLength=0.2,
                    )
                else:
                    prediction_vis = cv2.rectangle(
                        prediction_vis,
                        (
                            max(0, int(center[0]) - GRIPPER_GRASP_INNER_DISTANCE // 2),
                            max(0, int(center[1]) - GRIPPER_GRASP_WIDTH_PIXEL // 2),
                        ),
                        (
                            min(
                                prediction_vis.shape[1],
                                int(center[0]) + GRIPPER_GRASP_INNER_DISTANCE_PIXEL // 2,
                            ),
                            min(
                                prediction_vis.shape[0],
                                int(center[1]) + GRIPPER_GRASP_WIDTH_PIXEL // 2,
                            ),
                        ),
                        (100, 255, 0),
                        1,
                    )
                    prediction_vis = cv2.rectangle(
                        prediction_vis,
                        (
                            max(0, int(center[0]) - GRIPPER_GRASP_OUTER_DISTANCE_PIXEL // 2),
                            max(0, int(center[1]) - GRIPPER_GRASP_SAFE_WIDTH_PIXEL // 2),
                        ),
                        (
                            min(
                                prediction_vis.shape[1],
                                int(center[0]) + GRIPPER_GRASP_OUTER_DISTANCE_PIXEL // 2,
                            ),
                            min(
                                prediction_vis.shape[0],
                                int(center[1]) + GRIPPER_GRASP_SAFE_WIDTH_PIXEL // 2,
                            ),
                        ),
                        (100, 100, 155),
                        1,
                    )
            background_image = utils.rotate(color_heightmap, rotate_idx * (360.0 / num_rotations))
            prediction_vis = (
                0.5 * cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR) + 0.5 * prediction_vis
            ).astype(np.uint8)
            if tmp_row_canvas is None:
                tmp_row_canvas = prediction_vis
            else:
                tmp_row_canvas = np.concatenate((tmp_row_canvas, prediction_vis), axis=1)
        if canvas is None:
            canvas = tmp_row_canvas
        else:
            canvas = np.concatenate((canvas, tmp_row_canvas), axis=0)

    return canvas


class LifelongTrainer:
    def __init__(self, args):
        self.params = {
            "lr": args.lr,
            "batch_size": args.batch_size,
            "t_0": args.t_0,  # CosineAnnealing, start  0 4 12 28
            "t_mult": args.t_mult,  # CosineAnnealing, period 4 8 16
            "eta_min": args.eta_min,  # CosineAnnealing, minimum lr
            "epochs": args.epochs,  # CosineAnnealing, should end before warm start
            "loss_beta": args.loss_beta,
            "num_rotation": args.num_rotation,
            "ratio": args.ratio,
        }

        self.dataset_root = args.dataset_root
        self.pretrained_model = args.pretrained_model
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        log_dir_path = self.dataset_root[0: self.dataset_root.rfind('/')]
        self.log_dir = os.path.join(log_dir_path, "runs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        timestamp_value = datetime.datetime.fromtimestamp(time.time())
        time_name = timestamp_value.strftime("%Y-%m-%d-%H-%M")
        self.log_dir = os.path.join(self.log_dir, time_name)
        self.tb_logger = SummaryWriter(self.log_dir)
        self.logger = log_utils.setup_logger(self.log_dir, "Lifelong")

    def main(self):
        model = PushNet(True)
        # model = reinforcement_net(True)
        model = model.to(self.device)
        criterion = torch.nn.SmoothL1Loss(beta=self.params["loss_beta"], reduction="none")
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            momentum=0.9,
            weight_decay=2e-5,
        )
        # criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        # optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.params["t_0"],
            T_mult=self.params["t_mult"],
            eta_min=self.params["eta_min"],
            last_epoch=-1,
            verbose=False,
        )
        start_epoch = 0

        if self.pretrained_model is not None:
            checkpoint = torch.load(self.pretrained_model)
            model.load_state_dict(checkpoint["model"], strict=False)
            # optimizer.load_state_dict(checkpoint["optimizer"])
            # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            # start_epoch = checkpoint["epoch"] + 1
            # prev_params = checkpoint["params"]

        self.logger.info(f"Hyperparameters: {self.params}")
        if self.pretrained_model is not None:
            self.logger.info(f"Start from the pretrained model: {self.pretrained_model}")
            # self.logger.info(f"Previous Hyperparameters: {prev_params}")

        data_loader_train = self._get_data_loader(self.params["batch_size"], self.params["ratio"], shuffle=True)

        for epoch in range(start_epoch, self.params["epochs"]):
            # warmup start
            if epoch < 0:
                warmup_factor = 0.001
                warmup_iters = min(1000, len(data_loader_train) - 1)
                current_lr_scheduler = torch_utils.warmup_lr_scheduler(
                    optimizer, warmup_iters, warmup_factor
                )
            else:
                current_lr_scheduler = lr_scheduler

            train_loss = self._train_one_epoch(
                model, criterion, optimizer, data_loader_train, current_lr_scheduler, epoch,
            )

            if epoch % 2 == 0 or (self.params["epochs"] - epoch) < 2:
                save_state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "params": self.params,
                }
                torch.save(save_state, os.path.join(self.log_dir, f"lifelong_model-{epoch}.pth"))

            self.tb_logger.add_scalars("Epoch_Loss", {"train": train_loss}, epoch)
            self.tb_logger.flush()

        self.tb_logger.add_hparams(self.params, {"hparam/train": train_loss})
        self.logger.info("Training completed!")

    def _train_one_epoch(
        self, model, criterion, optimizer, data_loader, lr_scheduler, epoch, print_freq=50,
    ):
        model.train()
        metric_logger = log_utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", log_utils.SmoothedValue(window_size=1, fmt="{value:.12f}"))
        metric_logger.add_meter("loss", log_utils.SmoothedValue())
        metric_logger.add_meter("error", log_utils.SmoothedValue())
        header = "Epoch: [{}]".format(epoch)
        losses = []
        n_iter = 0
        total_iters = len(data_loader)

        for (
            mask_images,
            target_images,
            weight_images,
            best_locs
            # for (
            #     color_images,
            #     depth_images,
            #     help_images,
            #     target_images,
            #     weight_images,
            #     best_loc,
        ) in metric_logger.log_every(data_loader, print_freq, self.logger, header):
            # color_images = color_images.to(self.device, non_blocking=True)
            # depth_images = depth_images.to(self.device, non_blocking=True)
            # help_images = help_images.to(self.device, non_blocking=True)
            mask_images = mask_images.to(self.device, non_blocking=True)
            target_images = target_images.to(self.device, non_blocking=True)
            weight_images = weight_images.to(self.device, non_blocking=True)

            output_prob = model(mask_images)
            # output_prob = model(color_images, depth_images, help_images)
            # output_prob = model(
            #     color_images, depth_images, input_help_data=None, use_push=True, push_only=True
            # )

            errors = 0
            for i in range(best_locs.size(0)):
                error = (
                    output_prob[i, 0, best_locs[i][0], best_locs[i][1]]
                    - target_images[i, 0, best_locs[i][0], best_locs[i][1]]
                )
                error = error.abs()
                errors += error
            error = errors / best_locs.size(0)

            loss = criterion(output_prob, target_images) * weight_images
            loss = loss.sum() / target_images.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log
            log_loss = loss.item()
            log_lr = optimizer.param_groups[0]["lr"]
            log_error = error.item()
            metric_logger.update(loss=log_loss, lr=log_lr, error=log_error)
            self.tb_logger.add_scalar("Step/Train/Loss", log_loss, total_iters * epoch + n_iter)
            self.tb_logger.add_scalar("Step/Train/Error", log_error, total_iters * epoch + n_iter)
            self.tb_logger.add_scalar("Step/LR", log_lr, total_iters * epoch + n_iter)
            losses.append(log_loss)

            if epoch == 0:
                lr_scheduler.step()
            n_iter += 1

        if epoch != 0:
            lr_scheduler.step(epoch)

        # color version
        # push_predictions = output_prob[0][0].cpu().detach().numpy()
        # color_img = ToPILImage()(color_images[0].cpu()).convert("RGB")
        # color_img = np.array(color_img)
        # color_img = color_img[:, :, ::-1].copy()
        # center = np.unravel_index(np.argmax(push_predictions), push_predictions.shape)
        # point_from = (int(center[1]), int(center[0]))
        # point_to = (int(center[1] + PUSH_DISTANCE_PIXEL), int(center[0]))
        # color_img = cv2.arrowedLine(
        #     color_img, point_from, point_to, (100, 255, 0), 2, tipLength=0.2,
        # )
        # cv2.imwrite(f"vis{epoch}_color.png", color_img)
        # mask version
        push_predictions = output_prob[0][0].cpu().detach().numpy()
        center = np.unravel_index(np.argmax(push_predictions), push_predictions.shape)
        push_predictions[push_predictions < 0] = 0
        push_predictions = push_predictions / np.max(push_predictions)
        push_predictions = np.clip(push_predictions, 0, 1)
        push_predictions = cv2.applyColorMap(
            (push_predictions * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        mask_image = ToPILImage()(mask_images[0].cpu())
        mask_image = np.array(mask_image)
        mask_image = np.array(mask_image[:, :, 1])
        point_from = (int(center[1]), int(center[0]))
        point_to = (int(center[1] + PUSH_DISTANCE_PIXEL), int(center[0]))
        mask_image = cv2.arrowedLine(mask_image, point_from, point_to, 200, 2, tipLength=0.2,)
        point_from = (int(best_locs[0][1]), int(best_locs[0][0]))
        point_to = (point_from[0] + PUSH_DISTANCE_PIXEL, point_from[1])
        if torch.max(target_images[0]) >= 1:
            mask_image = cv2.arrowedLine(mask_image, point_from, point_to, 160, 3, tipLength=0.1,)
        else:
            mask_image = cv2.arrowedLine(mask_image, point_from, point_to, 100, 2, tipLength=0.1,)
        prediction_vis = (
            0.5 * cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR) + 0.5 * push_predictions
        ).astype(np.uint8)
        cv2.imwrite(f"vis{epoch}_mask.png", prediction_vis)

        return sum(losses) / len(losses)

    def _get_data_loader(self, batch_size, ratio=1, shuffle=False):
        """Get data loader."""
        dataset = LifelongDataset(self.dataset_root, ratio)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False
        )

        return data_loader


if __name__ == "__main__":
    args = parse_args()
    trainer = LifelongTrainer(args)
    trainer.main()

