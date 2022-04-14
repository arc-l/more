from torch.utils.data.sampler import Sampler
import os
import math
import re
import numpy as np
import torch
import torch.utils.data
import cv2
import imutils
from torchvision.transforms import functional as TF
from PIL import Image
import random
from constants import (
    IMAGE_OBJ_CROP_SIZE,
    IMAGE_SIZE,
    WORKSPACE_LIMITS,
    PIXEL_SIZE,
    PUSH_Q,
    GRASP_Q,
    COLOR_MEAN,
    COLOR_STD,
    DEPTH_MEAN,
    DEPTH_STD,
    BINARY_IMAGE_MEAN,
    BINARY_IMAGE_STD,
    BINARY_OBJ_MEAN,
    BINARY_OBJ_STD,
    DEPTH_MIN,
    PUSH_DISTANCE,
    GRIPPER_PUSH_RADIUS_PIXEL,
    GRIPPER_PUSH_RADIUS_SAFE_PIXEL,
    GRIPPER_GRASP_OUTER_DISTANCE_PIXEL,
    GRIPPER_GRASP_INNER_DISTANCE_PIXEL,
    IMAGE_PAD_WIDTH,
    PUSH_DISTANCE_PIXEL,
    GRIPPER_GRASP_WIDTH_PIXEL,
    NUM_ROTATION,
    IMAGE_PAD_DIFF,
)
from math import atan2, cos, sin, sqrt, pi, degrees
import glob
import pandas as pd
import utils


class LifelongEvalDataset(torch.utils.data.Dataset):
    """For lifelong learning"""

    def __init__(self, env, actions, mask_image, is_real=False):
        # relabel
        if is_real:
            mask_image = utils.relabel_mask_real(mask_image)
        else:
            mask_image = utils.relabel_mask(env, mask_image)
        # focus on target, so make one extra channel
        target_mask_img = np.zeros_like(mask_image, dtype=np.uint8)
        target_mask_img[mask_image == 255] = 255
        mask_heightmap = np.dstack((target_mask_img, mask_image))
        mask_heightmap_pad = np.pad(
            mask_heightmap,
            ((IMAGE_PAD_WIDTH, IMAGE_PAD_WIDTH), (IMAGE_PAD_WIDTH, IMAGE_PAD_WIDTH), (0, 0)),
            "constant",
            constant_values=0,
        )

        self.mask_heightmap_pad = mask_heightmap_pad
        self.actions = actions

    def __getitem__(self, idx):
        action = self.actions[idx]
        action_start = (action[0][1], action[0][0])
        action_end = (action[1][1], action[1][0])
        current = (
            action_end[0] - action_start[0],
            action_end[1] - action_start[1],
        )
        right = (1, 0)
        dot = (
            right[0] * current[0] + right[1] * current[1]
        )  # dot product between [x1, y1] and [x2, y2]
        det = right[0] * current[1] - right[1] * current[0]  # determinant
        rot_angle = math.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
        rot_angle = math.degrees(rot_angle)

        mask_heightmap_rotated = utils.rotate(self.mask_heightmap_pad, rot_angle, is_mask=True)
        input_image = mask_heightmap_rotated.astype(float) / 255
        input_image.shape = (
            input_image.shape[0],
            input_image.shape[1],
            input_image.shape[2],
        )

        with torch.no_grad():
            rot_angle = torch.tensor(rot_angle)
            input_data = torch.from_numpy(input_image.astype(np.float32)).permute(2, 0, 1)

        return rot_angle, input_data
        

    def __len__(self):
        return len(self.actions)



class LifelongDataset(torch.utils.data.Dataset):
    """For lifelong learning"""

    one_threshold = 0.9
    half_threshold = 0.8

    def __init__(self, root, ratio=1):

        self.root = root
        self.color_imgs = []
        self.depth_imgs = []
        self.mask_imgs = []
        self.best_locates = []
        self.labels = []
        self.weights = []
        self.kernel_collision = np.ones(
            (GRIPPER_PUSH_RADIUS_PIXEL * 2, GRIPPER_GRASP_WIDTH_PIXEL), dtype=np.float32
        )
        self.kernel_right = np.zeros(
            (
                GRIPPER_PUSH_RADIUS_PIXEL * 2,
                (PUSH_DISTANCE_PIXEL + round(GRIPPER_GRASP_WIDTH_PIXEL / 2)) * 2,
            ),
            dtype=np.float32,
        )
        self.kernel_right[:, PUSH_DISTANCE_PIXEL + round(GRIPPER_GRASP_WIDTH_PIXEL / 2) :] = 1
        self.kernel_erode = np.ones((IMAGE_OBJ_CROP_SIZE, IMAGE_OBJ_CROP_SIZE))

        sub_roots = glob.glob(f"{root}/*")
        sub_roots = sorted(sub_roots, key=lambda r: r[-3:])
        if ratio == 1:
            sub_roots = sub_roots
        else:
            ratio = int(1 / ratio)
            temp = []
            for idx, sr in enumerate(sub_roots):
                if idx % ratio == 0:
                    temp.append(sr)
            sub_roots = temp

        num_cases = 0

        for sub_root in sub_roots:
            if "runs" in sub_root:
                continue
            
            # load all image files, sorting them to ensure that they are aligned
            color_imgs = list(sorted(glob.glob(os.path.join(sub_root, "mcts", "color", "*.color.png"))))
            if len(color_imgs) == 0:
                continue
            depth_imgs = list(sorted(glob.glob(os.path.join(sub_root, "mcts", "depth", "*.depth.png"))))
            masks_imgs = list(sorted(glob.glob(os.path.join(sub_root, "mcts", "mask", "*.mask.png"))))
            records = pd.read_csv(os.path.join(sub_root, "mcts", "records.csv"))

            right = (1, 0)
            label_adjust = []
            weight_adjust = []
            current_label_idx = -1
            for row in records.itertuples():
                action = list(re.split(r"\D+", row.action))
                action = [int(a) for a in action if a.isnumeric()]
                # if it is invalid move
                if np.any(np.array(action) > IMAGE_SIZE - 1) or np.any(np.array(action) < 0):
                    continue
                action_start = (action[0], action[1])
                action_end = (action[2], action[3])
                current = (
                    action_end[0] - action_start[0],
                    action_end[1] - action_start[1],
                )
                dot = (
                    right[0] * current[0] + right[1] * current[1]
                )  # dot product between [x1, y1] and [x2, y2]
                det = right[0] * current[1] - right[1] * current[0]  # determinant
                rot_angle = math.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
                rot_angle = math.degrees(rot_angle)
                # if row.num_visits < 5:
                #     continue
                self.best_locates.append(
                    [rot_angle, action_start[1], action_start[0], action_end[1], action_end[0]]
                )
                self.color_imgs.append(color_imgs[row.image_idx])
                self.depth_imgs.append(depth_imgs[row.image_idx])
                self.mask_imgs.append(masks_imgs[row.image_idx])
                label = float(row.label)
                self.labels.append(label)
                weigiht = row.num_visits if row.num_visits < 50 else 50
                self.weights.append(weigiht)

                # 2 for the best, 1 for good, 0.5 for ok, others are 0
            #     if current_label_idx == -1:
            #         current_label_idx = row.image_idx
            #         label_adjust = [label]
            #     elif current_label_idx != row.image_idx:
            #         label_adjust = self.adjust(label_adjust)
            #         self.labels.extend(label_adjust)
            #         current_label_idx = row.image_idx
            #         label_adjust = [label]
            #     else:
            #         label_adjust.append(label)
            # if len(label_adjust) > 0:
            #     label_adjust = self.adjust(label_adjust)
            #     self.labels.extend(label_adjust)

            num_cases += 1
        print(
            sum(np.array(self.labels) >= 1),
            sum(np.array(self.labels) >= 0.5),
            sum(np.array(self.labels) < 0.5),
        )

        print(f"Total image used: {len(self.color_imgs)} and total test cases used: {num_cases}")

        assert len(self.color_imgs) == len(self.labels)

    def adjust_label(self, label_adjust):
        label_adjust = np.array(label_adjust)
        one_idx = np.where(label_adjust > np.quantile(label_adjust, LifelongDataset.one_threshold))
        half_idx = np.where(
            label_adjust > np.quantile(label_adjust, LifelongDataset.half_threshold)
        )
        mean_idx = np.where(label_adjust < np.mean(label_adjust[label_adjust != 2]))
        best_idx = np.argmax(label_adjust)
        label_adjust[:] = 0
        label_adjust[half_idx] = 0.5
        label_adjust[one_idx] = 1
        label_adjust[mean_idx] = 0
        label_adjust[best_idx] = 2
        label_adjust = list(label_adjust)
        return label_adjust

    def adjust_weight(self, label, weight_adjust):
        label = np.array(label)
        one_idx = np.where(label > np.quantile(label, LifelongDataset.one_threshold))
        half_idx = np.where(
            label > np.quantile(label, LifelongDataset.half_threshold)
        )
        mean_idx = np.where(label < np.mean(label[label != 2]))
        best_idx = np.argmax(label)
        label_adjust[:] = 0
        label_adjust[half_idx] = 0.5
        label_adjust[one_idx] = 1
        label_adjust[mean_idx] = 0
        label_adjust[best_idx] = 2
        label_adjust = list(label_adjust)
        return label_adjust

    def __getitem__(self, idx):
        # color image input
        # color_img = cv2.imread(self.color_imgs[idx])
        # color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        # depth image input
        # depth_img = cv2.imread(self.depth_imgs[idx], cv2.IMREAD_UNCHANGED)
        # depth_img = depth_img.astype(np.float32) / 100000  # translate to meters

        # mask image input
        mask_img = cv2.imread(self.mask_imgs[idx], cv2.IMREAD_UNCHANGED)
        # mask_id = np.unique(mask_img)
        # diff = -50
        # for mask_i in mask_id:
        #     if mask_i != 255 and mask_i != 0:
        #         mask_img[mask_img == mask_i] = mask_i + diff
        #         diff += 10

        rot_angle = self.best_locates[idx][0]
        action_start = (self.best_locates[idx][1], self.best_locates[idx][2])
        label = self.labels[idx]
        weight = self.weights[idx]
        # weight = self.weights[idx] * 10 if label >= 1 else 1
        # if self.weights[idx] > 10 or label >= 1:
        #     weight = self.weights[idx] * self.weights[idx] / 500 * 10
        # else:
        #     weight = self.weights[idx] * self.weights[idx] / 500

        # target image
        target_img = np.zeros_like(mask_img, dtype=np.float32)
        # target_img[
        #     self.best_locates[idx][1] - 3 : self.best_locates[idx][1] + 4,
        #     self.best_locates[idx][2] - 3 : self.best_locates[idx][2] + 4,
        # ] = (self.labels[idx] / 4 if self.labels[idx] > 0 else 0)
        # target_img[
        #     self.best_locates[idx][1] - 1 : self.best_locates[idx][1] + 2,
        #     self.best_locates[idx][2] - 1 : self.best_locates[idx][2] + 2,
        # ] = (self.labels[idx] / 2 if self.labels[idx] > 0 else 0)
        # target_img[self.best_locates[idx][1], self.best_locates[idx][2]] = self.labels[idx]
        # weight
        weight_img = np.zeros_like(mask_img, dtype=np.float32)
        # weight_img[
        #     self.best_locates[idx][1] - 3 : self.best_locates[idx][1] + 4,
        #     self.best_locates[idx][2] - 3 : self.best_locates[idx][2] + 4,
        # ] = 1
        # weight = self.weights[idx]
        # weight_img[self.best_locates[idx][1], self.best_locates[idx][2]] = (
        #     weight * weight / 500
        # )
        # weight_img[self.best_locates[idx][1], self.best_locates[idx][2]] = (
        #     100 if self.labels[idx] > 0 else 1
        # )
        # Post-process, collision checking
        # target_invalid = np.logical_and(weight_img > 0, depth_img > DEPTH_MIN)
        # target_img[target_invalid] = 0
        # weight_img[target_invalid] = 0.1
        # target_invalid = cv2.filter2D(depth_img, -1, self.kernel_collision)
        # target_img[(target_invalid > DEPTH_MIN)] = 0
        # weight_img[(target_invalid > DEPTH_MIN)] = 0.1

        # color_img_pil = Image.fromarray(color_img)
        # depth_img_pil = Image.fromarray(depth_img)

        # focus on target, so make one extra channel
        target_mask_img = np.zeros_like(mask_img, dtype=np.uint8)
        target_mask_img[mask_img == 255] = 255
        mask_img = np.dstack((target_mask_img, mask_img))

        # mask_img_pil = Image.fromarray(mask_img)
        # target_img_pil = Image.fromarray(target_img)
        # weight_img_pil = Image.fromarray(weight_img)
        mask_img_pil = mask_img
        target_img_pil = target_img
        weight_img_pil = weight_img

        (
            mask_img_pil,
            target_img_pil,
            weight_img_pil,
            best_loc,
        ) = self.transforms(
            mask_img_pil, target_img_pil, weight_img_pil, rot_angle, action_start, label, weight
        )

        # return (
        #     color_img_pil,
        #     depth_img_pil,
        #     mask_img_pil,
        #     help_img_pil,
        #     target_img_pil,
        #     weight_img_pil,
        #     best_loc,
        # )
        return (
            mask_img_pil,
            target_img_pil,
            weight_img_pil,
            best_loc,
        )

    def __len__(self):
        return len(self.color_imgs)

    @torch.no_grad()
    def transforms(
        self,
        mask_heightmap,
        target_heightmap,
        weight_heightmap,
        rot_angle,
        action_start, 
        label, 
        weight,
    ):

        # Add extra padding (to handle rotations inside network)
        # color_heightmap_pad = TF.pad(
        #     color_heightmap, IMAGE_PAD_WIDTH, fill=0, padding_mode="constant"
        # )
        # depth_heightmap_pad = TF.pad(
        #     depth_heightmap, IMAGE_PAD_WIDTH, fill=0, padding_mode="constant"
        # )
        # mask_heightmap_pad = TF.pad(
        #     mask_heightmap, IMAGE_PAD_WIDTH, fill=0, padding_mode="constant"
        # )
        # target_heightmap_pad = TF.pad(
        #     target_heightmap, IMAGE_PAD_WIDTH, fill=0, padding_mode="constant"
        # )
        # weight_heightmap_pad = TF.pad(
        #     weight_heightmap, IMAGE_PAD_WIDTH, fill=0, padding_mode="constant"
        # )
        mask_heightmap_pad = np.pad(
            mask_heightmap, ((IMAGE_PAD_WIDTH, IMAGE_PAD_WIDTH), (IMAGE_PAD_WIDTH, IMAGE_PAD_WIDTH), (0, 0)), "constant", constant_values=0
        )
        target_heightmap_pad = np.pad(
            target_heightmap, IMAGE_PAD_WIDTH, "constant", constant_values=0
        )
        weight_heightmap_pad = np.pad(
            weight_heightmap, IMAGE_PAD_WIDTH, "constant", constant_values=0
        )

        # color_heightmap_pad = TF.rotate(color_heightmap_pad, rot_angle)
        # depth_heightmap_pad = TF.rotate(depth_heightmap_pad, rot_angle)
        # mask_heightmap_pad = TF.rotate(mask_heightmap_pad, rot_angle)
        mask_heightmap_pad = utils.rotate(mask_heightmap_pad, rot_angle, is_mask=True)
        # target_heightmap_pad = TF.rotate(target_heightmap_pad, rot_angle, resample=PIL.Image.BILINEAR)
        # weight_heightmap_pad = TF.rotate(weight_heightmap_pad, rot_angle, resample=PIL.Image.BILINEAR)

        # color_heightmap_pad = np.array(color_heightmap_pad)
        # depth_heightmap_pad = np.array(depth_heightmap_pad)
        # mask_heightmap_pad = np.array(mask_heightmap_pad)
        # target_heightmap_pad = np.array(target_heightmap_pad)
        # weight_heightmap_pad = np.array(weight_heightmap_pad)
        # help_heightmap_pad = np.zeros_like(color_heightmap_pad)

        action_start = (action_start[0] + IMAGE_PAD_WIDTH, action_start[1] + IMAGE_PAD_WIDTH)
        origin = target_heightmap_pad.shape
        origin = ((origin[0] - 1) / 2, (origin[1] - 1) / 2)
        new_action_start = utils.rotate_point(origin, action_start, math.radians(rot_angle))
        new_action_start = (round(new_action_start[0]), round(new_action_start[1]))
        best_loc = torch.tensor(new_action_start)

        # Post-process, make single pixel larger
        target_heightmap_pad[
            best_loc[0] - 1 : best_loc[0] + 2, best_loc[1] - 1 : best_loc[1] + 2,
        ] = label if label > 0 else 0
        target_heightmap_pad[best_loc[0], best_loc[1]] = label if label > 0 else 0
        weight_heightmap_pad[
            best_loc[0] - 1 : best_loc[0] + 2, best_loc[1] - 1 : best_loc[1] + 2,
        ] = weight * 0.5
        weight_heightmap_pad[best_loc[0], best_loc[1]] = weight
        # Post-process, collision
        target_invalid = cv2.filter2D(mask_heightmap_pad[:, :, 1], -1, self.kernel_collision)
        target_heightmap_pad[(target_invalid > 0)] = 0
        weight_heightmap_pad[(target_invalid > 0)] = 0.001
        # Post-process, point to right
        target_invalid = cv2.filter2D(mask_heightmap_pad[:, :, 1], -1, self.kernel_right)
        target_heightmap_pad[(target_invalid == 0)] = 0
        weight_heightmap_pad[(target_invalid == 0)] = 0.0001
        # if np.max(target_heightmap_pad) >= 1:
        #     cv2.imshow("weight", (weight_heightmap_pad * 200 + 20).astype(np.uint8))
        #     cv2.imshow(
        #         f"target-{np.max(target_heightmap_pad)}",
        #         (target_heightmap_pad * 100 + 20).astype(np.uint8),
        #     )
        #     cv2.imshow("mask", mask_heightmap_pad)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        # Helper
        # mask of target object
        # temp = cv2.cvtColor(color_heightmap_pad, cv2.COLOR_RGB2HSV)
        # mask = cv2.inRange(temp, TARGET_LOWER, TARGET_UPPER)
        # # mask of clearance of target object
        # target_erode = cv2.filter2D(mask, -1, self.kernel_erode)
        # clearance = np.zeros_like(mask)
        # clearance[
        #     np.logical_and(
        #         np.logical_and(target_erode > 0, mask == 0), depth_heightmap_pad < DEPTH_MIN
        #     )
        # ] = 255
        # cv2.imshow("mask", mask)
        # cv2.imshow("clearance", clearance)
        # cv2.imshow("action", action)
        # cv2.imshow("color", color_heightmap_pad)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # To tensor
        # color_heightmap_pad = TF.to_tensor(color_heightmap_pad)
        # depth_heightmap_pad = TF.to_tensor(depth_heightmap_pad)
        target_heightmap_pad = TF.to_tensor(target_heightmap_pad)
        weight_heightmap_pad = TF.to_tensor(weight_heightmap_pad)
        mask_heightmap_pad = TF.to_tensor(mask_heightmap_pad)
        # mask = TF.to_tensor(mask)
        # clearance = TF.to_tensor(clearance)

        # Normalize
        # color_heightmap_pad = TF.normalize(color_heightmap_pad, COLOR_MEAN, COLOR_STD, inplace=True)
        # depth_heightmap_pad = TF.normalize(depth_heightmap_pad, DEPTH_MEAN, DEPTH_STD, inplace=True)
        # mask = TF.normalize(mask, BINARY_IMAGE_MEAN[1], BINARY_IMAGE_STD[1], inplace=True)
        # clearance = TF.normalize(clearance, BINARY_IMAGE_MEAN[1], BINARY_IMAGE_STD[1], inplace=True)


        return (
            mask_heightmap_pad,
            target_heightmap_pad,
            weight_heightmap_pad,
            best_loc,
        )


class SegmentationDataset(torch.utils.data.Dataset):
    """
    Create segmentation dataset for training Mask R-CNN.
    One uses pre-defined color range to separate objects (assume the color in one image is unique).
    One directly reads masks.
    """

    def __init__(self, root, transforms, is_real=False, background=None):
        self.root = root
        self.transforms = transforms
        self.is_real = is_real
        # load all image files, sorting them to ensure that they are aligned
        self.color_imgs = list(sorted(os.listdir(os.path.join(root, "color-heightmaps"))))
        self.depth_imgs = list(sorted(os.listdir(os.path.join(root, "depth-heightmaps"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))
        self.background = background
        if self.background is not None:
            self.background = cv2.imread(background)

    def __getitem__(self, idx):
        # load images
        color_path = os.path.join(self.root, "color-heightmaps", self.color_imgs[idx])
        # depth_path = os.path.join(self.root, "depth-heightmaps", self.depth_imgs[idx])

        # color image input
        color_img = cv2.imread(color_path)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if self.background is not None:
            # random background
            color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
            # background = cv2.resize(self.background, color_img.shape[:2], interpolation=cv2.INTER_AREA)
            color_img[mask_img == 0, :] = self.background[mask_img == 0, :]
            color_img = color_img.astype(np.int16)
            for channel in range(color_img.shape[2]):  # R, G, B
                c_random = np.random.rand(1)
                c_random *= 30
                c_random -= 15
                c_random = c_random.astype(np.int16)
                color_img[mask_img == 0, channel] = color_img[mask_img == 0, channel] + c_random
            color_img = np.clip(color_img, 0, 255)
            color_img = color_img.astype(np.uint8)
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        # get masks
        masks = []
        labels = []
        if self.is_real:
            gray = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)
            gray = gray.astype(np.uint8)
            blurred = cv2.medianBlur(gray, 5)
            thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
            cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            cnts = imutils.grab_contours(cnts)
            for c in cnts:
                if cv2.contourArea(c) > 100:
                    mask = np.zeros(color_img.shape[:2], np.uint8)
                    cv2.drawContours(mask, [c], -1, (1), -1)
                    masks.append(mask)
                    # cv2.imshow('mask' + self.color_imgs[idx], mask)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
        else:
            for ci in np.unique(mask_img):
                if ci != 0:
                    mask = mask_img == ci
                    if np.sum((mask == True)) > 100:
                        masks.append(mask)
                        # NOTE: assume there is a single type of objects will have more than 1000 instances
                        labels.append(ci // 1000)

        num_objs = len(masks)
        if num_objs > 0:
            masks = np.stack(masks, axis=0)

        # get bounding box coordinates for each mask
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
            if xmin == xmax or ymin == ymax:
                num_objs = 0

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        if num_objs > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = torch.as_tensor([0], dtype=torch.float32)
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        num_objs = torch.tensor(num_objs)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["num_obj"] = num_objs

        if self.transforms is not None:
            # img, target = self.transforms(img, target)
            img, target = self.transforms(color_img, target)

        return img, target

    def __len__(self):
        # return len(self.imgs)
        return len(self.color_imgs)


class ForegroundDataset(torch.utils.data.Dataset):
    """
    Craete binary image, 1 means foreground, 0 means background.
    For grasp, we care about the center of object, while considering the clearance of gripper.
    For push, we know all pushs are from left to right.
    This labeling approach is the as the in the function get_neg of trainer.py
    """

    def __init__(self, root, num_rotations):
        self.root = root
        # load all image files, sorting them to ensure that they are aligned
        self.color_imgs = list(sorted(os.listdir(os.path.join(root, "color-heightmaps"))))
        self.depth_imgs = list(sorted(os.listdir(os.path.join(root, "depth-heightmaps"))))
        self.num_rotations = num_rotations
        self.push_large_kernel = np.ones((41, 41))  # hyperparamter
        self.push_small_kernel = np.ones((13, 13))  # hyperparamter
        self.grasp_kernel = np.ones((9, 9))  # hyperparamter
        self.post_grasp_kernel = np.zeros(
            (GRIPPER_GRASP_WIDTH_PIXEL, GRIPPER_GRASP_OUTER_DISTANCE_PIXEL)
        )
        diff = math.ceil(
            (GRIPPER_GRASP_OUTER_DISTANCE_PIXEL - GRIPPER_GRASP_INNER_DISTANCE_PIXEL) / 2
        )
        self.post_grasp_kernel[:, 0:diff] = 1  # left
        self.post_grasp_kernel[:, (GRIPPER_GRASP_OUTER_DISTANCE_PIXEL - diff) :] = 1  # right

    def __getitem__(self, idx):
        # load images
        color_path = os.path.join(self.root, "color-heightmaps", self.color_imgs[idx])
        depth_path = os.path.join(self.root, "depth-heightmaps", self.depth_imgs[idx])

        # color image input
        color_img = cv2.imread(color_path)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        color_img_pil = Image.fromarray(color_img)

        # depth image input
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_img = depth_img.astype(np.float32) / 100000  # translate to meters
        depth_img_pil = Image.fromarray(depth_img)

        # binary push image target, we need boundary and some extra
        push_depth_img = np.copy(depth_img)
        push_depth_img[push_depth_img <= DEPTH_MIN] = 0
        push_depth_img[push_depth_img > DEPTH_MIN] = 1
        push_depth_large = cv2.filter2D(push_depth_img, -1, self.push_large_kernel)
        push_depth_large[push_depth_large < 1] = 0
        push_depth_large[push_depth_large > 1] = 1
        push_depth_small = cv2.filter2D(push_depth_img, -1, self.push_small_kernel)
        push_depth_small[push_depth_small < 1] = 0
        push_depth_small[push_depth_small > 1] = 1
        push_depth_final = push_depth_large - push_depth_small
        push_depth_final[push_depth_final < 0] = 0
        # prepare q values
        push_depth_final[push_depth_final == 1] = PUSH_Q
        push_depth_final[push_depth_final == 0] = 0
        target_push_img_pil = Image.fromarray(push_depth_final)

        # binary grasp image target, we need center part
        grasp_depth_img = np.copy(depth_img)
        grasp_depth_img[grasp_depth_img <= DEPTH_MIN] = -100
        grasp_depth_img[grasp_depth_img > DEPTH_MIN] = 1
        grasp_depth = cv2.filter2D(grasp_depth_img, -1, self.grasp_kernel)
        grasp_depth[grasp_depth < 1] = 0
        grasp_depth[grasp_depth > 1] = 1
        # # focus on target
        # color_mask = cv2.cvtColor(color_img, cv2.COLOR_RGB2HSV)
        # color_mask = cv2.inRange(color_mask, TARGET_LOWER, TARGET_UPPER)
        # grasp_depth[color_mask != 255] = 0
        # prepare q values
        grasp_depth[grasp_depth == 1] = GRASP_Q
        grasp_depth[grasp_depth == 0] = 0
        target_grasp_img_pil = Image.fromarray(grasp_depth)

        color_img_pil, depth_img_pil, target_push_img_pil, target_grasp_img_pil = self.transforms(
            color_img_pil, depth_img_pil, target_push_img_pil, target_grasp_img_pil
        )

        return color_img_pil, depth_img_pil, target_push_img_pil, target_grasp_img_pil

    def __len__(self):
        return len(self.color_imgs)

    @torch.no_grad()
    def transforms(
        self, color_heightmap, depth_heightmap, target_push_heightmap, target_grasp_heightmap
    ):

        # Add extra padding (to handle rotations inside network)
        color_heightmap_pad = TF.pad(
            color_heightmap, IMAGE_PAD_WIDTH, fill=0, padding_mode="constant"
        )
        depth_heightmap_pad = TF.pad(
            depth_heightmap, IMAGE_PAD_WIDTH, fill=0, padding_mode="constant"
        )
        depth_heightmap_pad_push = TF.pad(
            depth_heightmap, IMAGE_PAD_WIDTH, fill=-1, padding_mode="constant"
        )
        target_push_heightmap_pad = TF.pad(
            target_push_heightmap, IMAGE_PAD_WIDTH, fill=0, padding_mode="constant"
        )
        target_grasp_heightmap_pad = TF.pad(
            target_grasp_heightmap, IMAGE_PAD_WIDTH, fill=0, padding_mode="constant"
        )

        # Random rotate
        # rotate_idx = random.randint(0, self.num_rotations - 1)
        # rotate_theta = rotate_idx * (360 / self.num_rotations)
        rotate_theta = random.random() * 360
        color_heightmap_pad = TF.rotate(color_heightmap_pad, rotate_theta)
        depth_heightmap_pad = TF.rotate(depth_heightmap_pad, rotate_theta)
        depth_heightmap_pad_push = TF.rotate(depth_heightmap_pad_push, rotate_theta)
        target_push_heightmap_pad = TF.rotate(target_push_heightmap_pad, rotate_theta)
        target_grasp_heightmap_pad = TF.rotate(target_grasp_heightmap_pad, rotate_theta)

        color_heightmap_pad = np.array(color_heightmap_pad)
        depth_heightmap_pad = np.array(depth_heightmap_pad)
        depth_heightmap_pad_push = np.array(depth_heightmap_pad_push)
        target_push_heightmap_pad = np.array(target_push_heightmap_pad)
        target_grasp_heightmap_pad = np.array(target_grasp_heightmap_pad)
        # Post process for pushing, only pixel has something on the right (based
        # on heightmap) will be 1, otherwise it will be a empty push, also if the
        # pushed place is empty
        x_y_idx = np.argwhere(target_push_heightmap_pad > 0)
        for idx in x_y_idx:
            x, y = tuple(idx)
            area = depth_heightmap_pad[
                max(0, x - math.ceil(GRIPPER_PUSH_RADIUS_PIXEL / 4)) : min(
                    depth_heightmap_pad.shape[0], x + math.ceil(GRIPPER_PUSH_RADIUS_PIXEL / 4) + 1
                ),
                min(depth_heightmap_pad.shape[1], y + GRIPPER_PUSH_RADIUS_SAFE_PIXEL) : min(
                    depth_heightmap_pad.shape[1], y + math.ceil(PUSH_DISTANCE_PIXEL / 2) + 1
                ),
            ]
            if np.sum(area > DEPTH_MIN) == 0:
                target_push_heightmap_pad[x, y] = 0
            else:
                area = depth_heightmap_pad_push[
                    max(0, x - math.ceil(GRIPPER_PUSH_RADIUS_PIXEL / 2)) : min(
                        depth_heightmap_pad_push.shape[0],
                        x + math.ceil(GRIPPER_PUSH_RADIUS_PIXEL / 2) + 1,
                    ),
                    min(
                        depth_heightmap_pad_push.shape[1] - 1,
                        y + PUSH_DISTANCE_PIXEL + math.ceil(0.05 / PIXEL_SIZE),
                    ),
                ]  # 65 is a hyperparameters, (push) + 5 cm (close to limits)
                if np.sum(area < 0) > 0:  # out of the workspace
                    target_push_heightmap_pad[x, y] = 0
        # Post process for grasping, only pixel has clearance on the left/right (based on heightmap) will be 1
        x_y_idx = np.argwhere(target_grasp_heightmap_pad > 0)
        for idx in x_y_idx:
            x, y = tuple(idx)
            left_area = depth_heightmap_pad[
                max(0, x - math.ceil(GRIPPER_GRASP_WIDTH_PIXEL / 2)) : min(
                    depth_heightmap_pad.shape[0], x + math.ceil(GRIPPER_GRASP_WIDTH_PIXEL / 2) + 1
                ),
                max(0, y - math.ceil(GRIPPER_GRASP_OUTER_DISTANCE_PIXEL / 2)) : max(
                    0, y - math.ceil(GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 2) + 1
                ),
            ]
            right_area = depth_heightmap_pad[
                max(0, x - math.ceil(GRIPPER_GRASP_WIDTH_PIXEL / 2)) : min(
                    depth_heightmap_pad.shape[0], x + math.ceil(GRIPPER_GRASP_WIDTH_PIXEL / 2) + 1
                ),
                min(
                    depth_heightmap_pad.shape[1] - 1,
                    y + math.ceil(GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 2),
                ) : min(
                    depth_heightmap_pad.shape[1],
                    y + math.ceil(GRIPPER_GRASP_OUTER_DISTANCE_PIXEL / 2) + 1,
                ),
            ]
            if (
                np.sum(left_area > DEPTH_MIN) > 0
                and np.sum((left_area - depth_heightmap_pad[x, y]) > -0.05)
                > 0  # 0.05 cm is a hyperparameter
            ) or (
                np.sum(right_area > DEPTH_MIN) > 0
                and np.sum((right_area - depth_heightmap_pad[x, y]) > -0.05) > 0
            ):
                target_grasp_heightmap_pad[x, y] = 0

        # To tensor
        color_heightmap_pad = TF.to_tensor(color_heightmap_pad)
        depth_heightmap_pad = TF.to_tensor(depth_heightmap_pad)
        target_push_heightmap_pad = TF.to_tensor(target_push_heightmap_pad)
        target_grasp_heightmap_pad = TF.to_tensor(target_grasp_heightmap_pad)

        # Normalize
        color_heightmap_pad = TF.normalize(color_heightmap_pad, COLOR_MEAN, COLOR_STD, inplace=True)
        depth_heightmap_pad = TF.normalize(depth_heightmap_pad, DEPTH_MEAN, DEPTH_STD, inplace=True)

        return (
            color_heightmap_pad,
            depth_heightmap_pad,
            target_push_heightmap_pad,
            target_grasp_heightmap_pad,
        )


class PushPredictionMultiDatasetEvaluation(torch.utils.data.Dataset):
    """
    Push Prediction Dataset for Evaluation
    Input: Image, Action (x, y), Pose (x, y)
    Output: Diff_x, Diff_y, Diff_angle
    """

    def __init__(self, depth_imgs, actions, poses, binary_objs):
        self.distance = PUSH_DISTANCE
        self.workspace_limits = WORKSPACE_LIMITS
        self.heightmap_resolution = PIXEL_SIZE

        self.prev_depth_imgs = []
        self.prev_poses = []
        self.actions = []
        self.binary_objs = []
        # print("Total files", len(depth_imgs), len(actions), len(poses))

        for i in range(len(actions)):
            self.prev_depth_imgs.append(
                depth_imgs[i][IMAGE_PAD_WIDTH:IMAGE_PAD_DIFF, IMAGE_PAD_WIDTH:IMAGE_PAD_DIFF]
            )
            self.prev_poses.append(poses[i])
            self.actions.append(actions[i])
            self.binary_objs.append(binary_objs[i])

        # print("Used files", len(self.prev_depth_imgs), len(self.prev_poses), len(self.actions), len(self.binary_objs))
        assert (
            len(
                set(
                    [
                        len(self.prev_depth_imgs),
                        len(self.prev_poses),
                        len(self.actions),
                        len(self.binary_objs),
                    ]
                )
            )
            == 1
        )

    def __getitem__(self, idx):
        # depth image input
        prev_depth_img = self.prev_depth_imgs[idx]
        # number of objects
        num_obj = len(self.prev_poses[idx])
        # poses
        prev_poses = self.prev_poses[idx]
        # action
        action_start = self.actions[idx]
        action_end = np.array([action_start[0] + self.distance / PIXEL_SIZE, action_start[1]])

        # prev binary depth binary
        # obj
        prev_depth_binary_img_obj = np.copy(prev_depth_img)
        prev_depth_binary_img_obj[prev_depth_binary_img_obj <= DEPTH_MIN] = 0
        prev_depth_binary_img_obj[prev_depth_binary_img_obj > DEPTH_MIN] = 1
        temp = np.zeros((680, 680))
        temp[228:452, 228:452] = prev_depth_binary_img_obj
        prev_depth_binary_img_obj = temp[
            int(action_start[0] + 228) - 40 : int(action_start[0] + 228) + 184,
            int(action_start[1] + 228) - 112 : int(action_start[1] + 228) + 112,
        ]
        # action
        prev_depth_binary_img_action = np.zeros_like(prev_depth_img)
        prev_depth_binary_img_action[
            int(action_start[0]) : int(action_end[0]),
            int(action_start[1])
            - GRIPPER_PUSH_RADIUS_PIXEL : int(action_start[1])
            + GRIPPER_PUSH_RADIUS_PIXEL
            + 1,
        ] = 1
        temp = np.zeros((680, 680))
        temp[228:452, 228:452] = prev_depth_binary_img_action
        prev_depth_binary_img_action = temp[
            int(action_start[0] + 228) - 40 : int(action_start[0] + 228) + 184,
            int(action_start[1] + 228) - 112 : int(action_start[1] + 228) + 112,
        ]

        binary_objs = self.binary_objs[idx]
        temp = np.zeros_like(binary_objs[:, :, 0:1])

        # centralize
        action_start_ori = action_start.copy()
        action_end_ori = action_end.copy()
        action_start[0] -= 40
        action_start[1] -= 112
        for pi in range(num_obj):
            prev_poses[pi] = prev_poses[pi] - action_start
        prev_poses = prev_poses.flatten()
        prev_poses = torch.tensor(prev_poses, dtype=torch.float32)
        action = torch.tensor(
            [40.0, 112.0, 40.0 + self.distance / PIXEL_SIZE, 112.0], dtype=torch.float32
        )

        used_binary_img, binary_objs_total = self.transforms(
            prev_depth_binary_img_obj, prev_depth_binary_img_action, binary_objs
        )

        return (
            prev_poses,
            action,
            action_start_ori,
            action_end_ori,
            used_binary_img,
            binary_objs_total,
            num_obj,
        )

    def __len__(self):
        return len(self.actions)

    @torch.no_grad()
    def transforms(self, prev_depth_binary_img_obj, prev_depth_binary_img_action, binary_objs):
        prev_depth_binary_img_obj = TF.to_tensor(prev_depth_binary_img_obj)
        prev_depth_binary_img_action = TF.to_tensor(prev_depth_binary_img_action)
        used_binary_img = torch.cat(
            (prev_depth_binary_img_obj, prev_depth_binary_img_action), dim=0
        )
        used_binary_img = TF.normalize(
            used_binary_img, BINARY_IMAGE_MEAN, BINARY_IMAGE_STD, inplace=True
        )
        binary_objs_total = TF.to_tensor(binary_objs)
        current_binary_mean = BINARY_OBJ_MEAN * binary_objs_total.size(0)
        current_binary_std = BINARY_OBJ_STD * binary_objs_total.size(0)
        binary_objs_total = TF.normalize(
            binary_objs_total, current_binary_mean, current_binary_std, inplace=True
        )

        return used_binary_img, binary_objs_total


class ClusterRandomSampler(Sampler):
    """Takes a dataset with cluster_indices property, cuts it into batch-sized chunks
    Drops the extra items, not fitting into exact batches
    Arguments:
        data_source (Dataset): a Dataset to sample from. Should have a cluster_indices property
        batch_size (int): a batch size that you would like to use later with Dataloader class
        shuffle (bool): whether to shuffle the data or not
    """

    def __init__(self, data_source, batch_size, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_after_batch = 0
        for _, cluster_indices in self.data_source.cluster_indices.items():
            self.num_after_batch += len(cluster_indices) // self.batch_size * self.batch_size

    def flatten_list(self, lst):
        return [item for sublist in lst for item in sublist]

    def __iter__(self):

        batch_lists = []
        for _, cluster_indices in self.data_source.cluster_indices.items():
            if self.shuffle:
                random.shuffle(cluster_indices)
            batches = [
                cluster_indices[i : i + self.batch_size]
                for i in range(0, len(cluster_indices), self.batch_size)
            ]
            # filter our the shorter batches
            if len(batches[-1]) != self.batch_size:
                batch_lists.append(batches[:-1])
            else:
                batch_lists.append(batches)

        # flatten lists and shuffle the batches if necessary
        # this works on batch level
        lst = self.flatten_list(batch_lists)
        if self.shuffle:
            random.shuffle(lst)
        # final flatten  - produce flat list of indexes
        lst = self.flatten_list(lst)
        return iter(lst)

    def __len__(self):
        return self.num_after_batch


class PushPredictionMultiDataset(torch.utils.data.Dataset):
    """
    Push Prediction Dataset for training Push prediction network.
    The push distance is fixed, could be 5 or 10 cm.
    Track objects by color, so we assume each object has a unique color, however, this constraint does not needed in evalution.
    Input: Image, Action (x, y), Pose (x, y)
    Output: Diff_x, Diff_y, Diff_angle
    """

    def __init__(self, root, distance, is_padding=False, cutoff=None):
        self.root = root
        self.is_padding = is_padding
        # load all image files, sorting them to ensure that they are aligned
        prev_color_imgs = list(sorted(os.listdir(os.path.join(root, "prev-color-heightmaps"))))
        prev_depth_imgs = list(sorted(os.listdir(os.path.join(root, "prev-depth-heightmaps"))))
        prev_poses = list(sorted(os.listdir(os.path.join(root, "prev-poses"))))
        next_color_imgs = list(sorted(os.listdir(os.path.join(root, "next-color-heightmaps"))))
        next_depth_imgs = list(sorted(os.listdir(os.path.join(root, "next-depth-heightmaps"))))
        next_poses = list(sorted(os.listdir(os.path.join(root, "next-poses"))))
        actions = list(sorted(os.listdir(os.path.join(root, "actions"))))

        self.distance = distance
        self.workspace_limits = WORKSPACE_LIMITS
        self.heightmap_resolution = PIXEL_SIZE

        self.prev_color_imgs = []
        self.prev_depth_imgs = []
        self.prev_poses = []
        self.actions = []
        self.next_color_imgs = []
        self.next_depth_imgs = []
        self.next_poses = []
        self.cluster_indices = {}
        print(
            "Total files",
            len(prev_color_imgs),
            len(prev_depth_imgs),
            len(prev_poses),
            len(actions),
            len(next_color_imgs),
            len(next_depth_imgs),
            len(next_poses),
        )

        for i in range(len(actions)):
            assert (
                len(
                    set(
                        [
                            actions[i][:7],
                            prev_color_imgs[i][:7],
                            prev_depth_imgs[i][:7],
                            prev_poses[i][:7],
                            next_color_imgs[i][:7],
                            next_depth_imgs[i][:7],
                            next_poses[i][:7],
                        ]
                    )
                )
                == 1
            ), (
                actions[i][:7],
                prev_color_imgs[i][:7],
                prev_depth_imgs[i][:7],
                prev_poses[i][:7],
                next_color_imgs[i][:7],
                next_depth_imgs[i][:7],
                next_poses[i][:7],
            )
            if cutoff is not None:
                if int(actions[i][:7]) > cutoff:
                    break

            self.prev_color_imgs.append(prev_color_imgs[i])
            self.prev_depth_imgs.append(prev_depth_imgs[i])
            self.prev_poses.append(prev_poses[i])
            self.actions.append(actions[i])
            self.next_color_imgs.append(next_color_imgs[i])
            self.next_depth_imgs.append(next_depth_imgs[i])
            self.next_poses.append(next_poses[i])
            # create cluster indices, so the the data with same amount of object will be put together
            poses_path = os.path.join(self.root, "prev-poses", prev_poses[i])
            with open(poses_path, "r") as file:
                filedata = file.read()
                poses_str = filedata.split(" ")
                num_obj = len(poses_str) // 5
                if num_obj in self.cluster_indices:
                    self.cluster_indices[num_obj].append(len(self.prev_poses) - 1)
                else:
                    self.cluster_indices[num_obj] = [len(self.prev_poses) - 1]

        print(
            "Used files",
            len(self.prev_color_imgs),
            len(self.next_color_imgs),
            len(self.prev_depth_imgs),
            len(self.next_depth_imgs),
            len(self.prev_poses),
            len(self.next_poses),
            len(self.actions),
        )
        assert (
            len(
                set(
                    [
                        len(self.prev_color_imgs),
                        len(self.next_color_imgs),
                        len(self.prev_depth_imgs),
                        len(self.next_depth_imgs),
                        len(self.prev_poses),
                        len(self.next_poses),
                        len(self.actions),
                    ]
                )
            )
            == 1
        )

    def __getitem__(self, idx):
        # load data path
        prev_color_path = os.path.join(
            self.root, "prev-color-heightmaps", self.prev_color_imgs[idx]
        )
        prev_depth_path = os.path.join(
            self.root, "prev-depth-heightmaps", self.prev_depth_imgs[idx]
        )
        prev_poses_path = os.path.join(self.root, "prev-poses", self.prev_poses[idx])
        actions_path = os.path.join(self.root, "actions", self.actions[idx])
        next_color_path = os.path.join(
            self.root, "next-color-heightmaps", self.next_color_imgs[idx]
        )
        next_depth_path = os.path.join(
            self.root, "next-depth-heightmaps", self.next_depth_imgs[idx]
        )
        next_poses_path = os.path.join(self.root, "next-poses", self.next_poses[idx])

        # color image input
        prev_color_img = cv2.imread(prev_color_path)
        prev_color_img = cv2.cvtColor(prev_color_img, cv2.COLOR_BGR2RGB)
        next_color_img = cv2.imread(next_color_path)
        next_color_img = cv2.cvtColor(next_color_img, cv2.COLOR_BGR2RGB)

        # depth image input
        prev_depth_img = cv2.imread(prev_depth_path, cv2.IMREAD_UNCHANGED)
        prev_depth_img = prev_depth_img.astype(np.float32) / 100000  # translate to meters 100000
        next_depth_img = cv2.imread(next_depth_path, cv2.IMREAD_UNCHANGED)
        next_depth_img = next_depth_img.astype(np.float32) / 100000  # translate to meters 100000
        next_depth_img[next_depth_img < 0] = 0

        # poses
        with open(prev_poses_path, "r") as file:
            filedata = file.read()
            poses = filedata.split(" ")
            num_obj = len(poses) // 5
            prev_poses = []
            for pi in range(num_obj):
                x = (float(poses[pi * 5]) - self.workspace_limits[0][0]) / self.heightmap_resolution
                y = (
                    float(poses[pi * 5 + 1]) - self.workspace_limits[1][0]
                ) / self.heightmap_resolution
                angle_y = degrees(float(poses[pi * 5 + 4]))
                prev_poses.extend([x, y, angle_y])
            prev_poses = torch.tensor(prev_poses)
        with open(next_poses_path, "r") as file:
            filedata = file.read()
            poses = filedata.split(" ")
            assert len(poses) // 5 == num_obj
            next_poses = []
            for pi in range(num_obj):
                x = (float(poses[pi * 5]) - self.workspace_limits[0][0]) / self.heightmap_resolution
                y = (
                    float(poses[pi * 5 + 1]) - self.workspace_limits[1][0]
                ) / self.heightmap_resolution
                angle_y = degrees(float(poses[pi * 5 + 4]))
                next_poses.extend([x, y, angle_y])
            next_poses = torch.tensor(next_poses)

        # action
        with open(actions_path, "r") as file:
            filedata = file.read()
            x, y, _ = filedata.split(" ")
            x = (float(x) - self.workspace_limits[0][0]) / self.heightmap_resolution
            y = (float(y) - self.workspace_limits[1][0]) / self.heightmap_resolution
            action_start = torch.tensor([float(x), float(y)])
            action_end = torch.tensor([float(x + self.distance / PIXEL_SIZE), float(y)])

        # prev binary depth binary
        # obj
        prev_depth_binary_img_obj = np.copy(prev_depth_img)
        prev_depth_binary_img_obj[prev_depth_binary_img_obj <= DEPTH_MIN] = 0
        prev_depth_binary_img_obj[prev_depth_binary_img_obj > DEPTH_MIN] = 1
        temp = np.zeros((680, 680))
        temp[228:452, 228:452] = prev_depth_binary_img_obj
        prev_depth_binary_img_obj = temp[
            int(action_start[0] + 228) - 40 : int(action_start[0] + 228) + 184,
            int(action_start[1] + 228) - 112 : int(action_start[1] + 228) + 112,
        ]

        # action
        prev_depth_binary_img_action = np.zeros_like(prev_depth_img)
        prev_depth_binary_img_action[
            int(action_start[0]) : int(action_end[0]),
            int(action_start[1])
            - GRIPPER_PUSH_RADIUS_PIXEL : int(action_start[1])
            + GRIPPER_PUSH_RADIUS_PIXEL
            + 1,
        ] = 1
        temp = np.zeros((680, 680))
        temp[228:452, 228:452] = prev_depth_binary_img_action
        prev_depth_binary_img_action = temp[
            int(action_start[0] + 228) - 40 : int(action_start[0] + 228) + 184,
            int(action_start[1] + 228) - 112 : int(action_start[1] + 228) + 112,
        ]

        # TODO: assume pose in order of blue, green, brown, orange, yellow
        imgcolor = np.copy(prev_color_img)
        imgcolor = imgcolor.astype(np.uint8)
        temp = np.zeros((480, 480, 3), dtype=np.uint8)
        temp[128 : (480 - 128), 128 : (480 - 128), :] = imgcolor
        imgcolor = cv2.cvtColor(temp, cv2.COLOR_RGB2HSV)
        binary_objs = []
        for ci in range(num_obj):
            crop = imgcolor[
                int(prev_poses[ci * 3]) + 128 - 30 : int(prev_poses[ci * 3]) + 128 + 30,
                int(prev_poses[ci * 3 + 1]) + 128 - 30 : int(prev_poses[ci * 3 + 1]) + 128 + 30,
                :,
            ]
            assert crop.shape[0] == 60 and crop.shape[1] == 60, (
                self.prev_color_imgs[idx],
                crop.shape,
            )
            mask = cv2.inRange(crop, colors_lower[ci], colors_upper[ci])
            binary_objs.append(mask)

        # delta poses
        deltas = []
        for pi in range(num_obj):
            d_x = next_poses[pi * 3] - prev_poses[pi * 3]
            d_y = next_poses[pi * 3 + 1] - prev_poses[pi * 3 + 1]
            d_a = next_poses[pi * 3 + 2] - prev_poses[pi * 3 + 2]
            if d_a < -180:
                d_a = 360 + d_a
            elif d_a > 180:
                d_a = d_a - 360
            assert abs(d_a) < 120, (
                pi,
                d_a,
                self.prev_color_imgs[idx],
                self.next_color_imgs[idx],
                prev_poses,
                next_poses,
            )
            deltas.extend([d_x, d_y, d_a])
        deltas = torch.tensor(deltas, dtype=torch.float32)

        # centralize
        action_start_ori = torch.clone(action_start).detach()
        action_end_ori = torch.clone(action_end).detach()
        action_start[0] -= 40
        action_start[1] -= 112
        for pi in range(num_obj):
            prev_poses[pi * 3 : pi * 3 + 2] = prev_poses[pi * 3 : pi * 3 + 2] - action_start
            next_poses[pi * 3 : pi * 3 + 2] = next_poses[pi * 3 : pi * 3 + 2] - action_start
        prev_poses_no_angle = []
        for pi in range(num_obj):
            prev_poses_no_angle.extend([prev_poses[pi * 3], prev_poses[pi * 3 + 1]])
        next_poses_no_angle = []
        for pi in range(num_obj):
            next_poses_no_angle.extend([next_poses[pi * 3], next_poses[pi * 3 + 1]])
        prev_poses = torch.tensor(prev_poses_no_angle, dtype=torch.float32)
        next_poses = torch.tensor(next_poses_no_angle, dtype=torch.float32)
        action = torch.tensor([40.0, 112.0, 40.0 + self.distance / PIXEL_SIZE, 112.0])
        num_obj = torch.tensor(num_obj)

        (
            prev_color_img,
            prev_depth_img,
            next_color_img,
            next_depth_img,
            used_binary_img,
            binary_objs_total,
        ) = self.transforms(
            prev_color_img,
            prev_depth_img,
            next_color_img,
            next_depth_img,
            prev_depth_binary_img_obj,
            prev_depth_binary_img_action,
            binary_objs,
        )

        return (
            prev_color_img,
            prev_depth_img,
            next_color_img,
            next_depth_img,
            used_binary_img,
            prev_poses,
            next_poses,
            action,
            deltas,
            self.prev_color_imgs[idx],
            self.next_color_imgs[idx],
            action_start_ori,
            action_end_ori,
            binary_objs_total,
            num_obj,
        )

    def __len__(self):
        return len(self.actions)

    @torch.no_grad()
    def transforms(
        self,
        prev_color_img,
        prev_depth_img,
        next_color_img,
        next_depth_img,
        prev_depth_binary_img_obj,
        prev_depth_binary_img_action,
        binary_objs,
    ):
        # To tensor
        prev_color_img = TF.to_tensor(prev_color_img)
        prev_depth_img = TF.to_tensor(prev_depth_img)
        next_color_img = TF.to_tensor(next_color_img)
        next_depth_img = TF.to_tensor(next_depth_img)
        prev_depth_binary_img_obj = TF.to_tensor(prev_depth_binary_img_obj)
        prev_depth_binary_img_action = TF.to_tensor(prev_depth_binary_img_action)
        used_binary_img = torch.cat(
            (prev_depth_binary_img_obj, prev_depth_binary_img_action), dim=0
        )
        used_binary_img = TF.normalize(
            used_binary_img, BINARY_IMAGE_MEAN, BINARY_IMAGE_STD, inplace=True
        )
        binary_objs_total = TF.to_tensor(binary_objs[0])
        for ci in range(1, len(binary_objs)):
            temp = TF.to_tensor(binary_objs[ci])
            temp = TF.normalize(temp, BINARY_OBJ_MEAN, BINARY_OBJ_STD, inplace=True)
            binary_objs_total = torch.cat((binary_objs_total, temp), dim=0)

        # Normalize
        prev_color_img = TF.normalize(prev_color_img, COLOR_MEAN, COLOR_STD, inplace=True)
        next_color_img = TF.normalize(next_color_img, COLOR_MEAN, COLOR_STD, inplace=True)
        prev_depth_img = TF.normalize(prev_depth_img, DEPTH_MEAN, DEPTH_STD, inplace=True)
        next_depth_img = TF.normalize(next_depth_img, DEPTH_MEAN, DEPTH_STD, inplace=True)

        return (
            prev_color_img,
            prev_depth_img,
            next_color_img,
            next_depth_img,
            used_binary_img,
            binary_objs_total,
        )


def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)

    angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)


def getCenterOrientation(pts, img):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))

    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (
        cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0],
        cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0],
    )
    p2 = (
        cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0],
        cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0],
    )
    drawAxis(img, cntr, p1, (0, 255, 0), 1)
    drawAxis(img, cntr, p2, (255, 255, 0), 5)
    # angle = math.atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
    angle = atan2(eigenvectors[1, 1], eigenvectors[1, 0])  # orientation in radians
    return cntr[0], cntr[1], angle


def compute_mean_std(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=4, shuffle=False)
    color_mean = 0
    color_std = 0
    depth_mean = 0
    depth_std = 0
    num_samples = 0.0
    for color, depth, _, _ in loader:
        batch_samples = color.size(0)
        color = color.view(batch_samples, color.size(1), -1)
        color_mean += color.mean(2).sum(0)
        color_std += color.std(2).sum(0)
        depth = depth.view(batch_samples, depth.size(1), -1)
        depth_mean += depth.mean(2).sum(0)
        depth_std += depth.std(2).sum(0)
        num_samples += batch_samples
    color_mean /= num_samples
    color_std /= num_samples
    print(f"color mean: {color_mean}, color std: {color_std}")
    depth_mean /= num_samples
    depth_std /= num_samples
    print(f"depth mean: {depth_mean}, depth std: {depth_std}")

    # sampler = ClusterRandomSampler(dataset, batch_size=64)
    # data_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=64,
    #     sampler=sampler,
    #     shuffle=False,
    #     num_workers=4,
    #     drop_last=True,
    # )
    # binary_image_mean = 0
    # binary_image_std = 0
    # binary_obj_mean = 0
    # binary_obj_std = 0
    # num_samples = 0
    # for _, _, _, _, used_binary_img, _, _, _, _, _, _, _, _, binary_objs_total, _ in data_loader:
    #     batch_samples = used_binary_img.size(0)
    #     used_binary_img = used_binary_img.view(batch_samples, used_binary_img.size(1), -1)
    #     binary_image_mean += used_binary_img.mean(2).sum(0)
    #     binary_image_std += used_binary_img.std(2).sum(0)
    #     binary_objs_total = binary_objs_total.view(batch_samples, binary_objs_total.size(1), -1)
    #     binary_obj_mean += binary_objs_total.mean(2).mean(1).sum(0)
    #     binary_obj_std += binary_objs_total.std(2).mean(1).sum(0)
    #     num_samples += batch_samples
    # binary_image_mean /= num_samples
    # binary_image_std /= num_samples
    # print(f"binary image mean: {binary_image_mean}, binary image std: {binary_image_std}")
    # binary_obj_mean /= num_samples
    # binary_obj_std /= num_samples
    # print(f"binary obj mean: {binary_obj_mean}, binary obj std: {binary_obj_std}")


if __name__ == "__main__":
    dataset = ForegroundDataset("logs_image/foreground/data", NUM_ROTATION)
    # dataset = PushPredictionMultiDataset("logs_push/push-05/train", PUSH_DISTANCE)
    compute_mean_std(dataset)
