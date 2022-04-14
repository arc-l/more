import argparse
from collections import defaultdict
import math
import time
from constants import (
    COLOR_MEAN,
    COLOR_STD,
    DEPTH_MEAN,
    DEPTH_STD,
    GRASP_Q_GRASP_THRESHOLD,
    GRIPPER_GRASP_WIDTH_PIXEL,
    GRIPPER_PUSH_RADIUS_PIXEL,
    IMAGE_PAD_WIDTH,
    IS_REAL,
    NUM_ROTATION,
    PIXEL_SIZE,
    PUSH_DISTANCE_PIXEL,
    TARGET_LOWER,
    TARGET_UPPER,
    WORKSPACE_LIMITS,
)
import numpy as np
from mcts_utils import MCTSHelper
import random

import cv2
import torch

from environment_sim import Environment
from models import reinforcement_net, PushNet
from mcts_main import SeachCollector
import utils

from mcts_utils import _sampled_prediction_precise
from mcts_main import SeachCollector


@torch.no_grad()
def get_q(model, color_heightmap, depth_heightmap):
    color_heightmap_pad = np.copy(color_heightmap)
    depth_heightmap_pad = np.copy(depth_heightmap)

    # Add extra padding (to handle rotations inside network)
    color_heightmap_pad = np.pad(
        color_heightmap_pad,
        ((IMAGE_PAD_WIDTH, IMAGE_PAD_WIDTH), (IMAGE_PAD_WIDTH, IMAGE_PAD_WIDTH), (0, 0)),
        "constant",
        constant_values=0,
    )
    depth_heightmap_pad = np.pad(
        depth_heightmap_pad, IMAGE_PAD_WIDTH, "constant", constant_values=0
    )

    # Pre-process color image (scale and normalize)
    image_mean = COLOR_MEAN
    image_std = COLOR_STD
    input_color_image = color_heightmap_pad.astype(float) / 255
    for c in range(3):
        input_color_image[:, :, c] = (input_color_image[:, :, c] - image_mean[c]) / image_std[c]

    # Pre-process depth image (normalize)
    image_mean = DEPTH_MEAN
    image_std = DEPTH_STD
    depth_heightmap_pad.shape = (depth_heightmap_pad.shape[0], depth_heightmap_pad.shape[1], 1)
    input_depth_image = np.copy(depth_heightmap_pad)
    input_depth_image[:, :, 0] = (input_depth_image[:, :, 0] - image_mean[0]) / image_std[0]

    # Construct minibatch of size 1 (b,c,h,w)
    input_color_image.shape = (
        input_color_image.shape[0],
        input_color_image.shape[1],
        input_color_image.shape[2],
        1,
    )
    input_depth_image.shape = (
        input_depth_image.shape[0],
        input_depth_image.shape[1],
        input_depth_image.shape[2],
        1,
    )
    input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(3, 2, 0, 1)
    input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(3, 2, 0, 1)

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

    # temp = cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2HSV)
    # mask = cv2.inRange(temp, TARGET_LOWER, TARGET_UPPER)
    # mask_pad = np.pad(mask, IMAGE_PAD_WIDTH, "constant", constant_values=0)
    # mask_pad.shape = (
    #     mask_pad.shape[0],
    #     mask_pad.shape[1],
    #     mask_pad.shape[2],
    #     1,
    # )
    # mask_pad = torch.from_numpy(mask_pad.astype(np.float32)).permute(3, 2, 0, 1)

    # Pass input data through model
    # output_prob = model(input_color_data, input_depth_data, True, -1, use_push=True, push_only=True)
    output_prob = model(input_color_data, input_depth_data, True, -1)

    # Return Q values (and remove extra padding)
    for rotate_idx in range(len(output_prob)):
        # output = torch.sigmoid(output_prob[rotate_idx])
        output = output_prob[rotate_idx]
        if rotate_idx == 0:
            push_predictions = output.cpu().data.numpy()[
                :, 0, :, :,
            ]
        else:
            push_predictions = np.concatenate(
                (push_predictions, output.cpu().data.numpy()[:, 0, :, :,],), axis=0,
            )
    padding_width_start = IMAGE_PAD_WIDTH
    padding_width_end = push_predictions[0].shape[0] - IMAGE_PAD_WIDTH
    push_predictions = push_predictions[
        :, padding_width_start:padding_width_end, padding_width_start:padding_width_end
    ]

    best_pix_ind = np.unravel_index(np.argmax(push_predictions), push_predictions.shape)
    grasp_q_value = push_predictions[best_pix_ind]

    return grasp_q_value, best_pix_ind, push_predictions


def filter_prediction(mask_heightmap, push_predictions):
    kernel_collision = np.ones(
        (GRIPPER_PUSH_RADIUS_PIXEL * 2, GRIPPER_GRASP_WIDTH_PIXEL), dtype=np.float32
    )
    kernel_right = np.zeros(
        (
            GRIPPER_PUSH_RADIUS_PIXEL * 2,
            (PUSH_DISTANCE_PIXEL + round(GRIPPER_GRASP_WIDTH_PIXEL / 2)) * 2,
        ),
        dtype=np.float32,
    )
    kernel_right[:, PUSH_DISTANCE_PIXEL + round(GRIPPER_GRASP_WIDTH_PIXEL / 2) :] = 1

    num_rotations = push_predictions.shape[0]
    for canvas_row in range(int(num_rotations / 4)):
        for canvas_col in range(4):
            rotate_idx = canvas_row * 4 + canvas_col
            # rotate
            pred_pad = utils.rotate(
                push_predictions[rotate_idx], rotate_idx * (360.0 / num_rotations)
            )
            mask_pad = utils.rotate(
                mask_heightmap, rotate_idx * (360.0 / num_rotations), is_mask=True
            )
            # filter collision
            target_invalid = cv2.filter2D(mask_pad, -1, kernel_collision)
            pred_pad[(target_invalid > 0)] = 0
            # # filter point to right
            target_invalid = cv2.filter2D(mask_pad, -1, kernel_right)
            pred_pad[(target_invalid == 0)] = 0
            # rotate back
            pred_pad = utils.rotate(pred_pad, -rotate_idx * (360.0 / num_rotations))
            push_predictions[rotate_idx] = pred_pad
    return push_predictions


@torch.no_grad()
def sampled_prediction_precise(mcts_helper, env, model, color_image, mask_image):
    actions = mcts_helper.sample_actions(None, color_image, mask_image)

    out_q = _sampled_prediction_precise(env, model, actions, mask_image)
    print(out_q)

    final = actions[np.argmax(out_q)]
    return final[0], final[1]


def sampled_prediction(mcts_helper, env, color_image, mask_image, push_predictions):
    actions = mcts_helper.sample_actions(None, color_image, mask_image)
    right = (1, 0)
    new_push_predictions = np.zeros_like(push_predictions)

    for action in actions:
        action_start = (action[0][1], action[0][0])
        action_end = (action[1][1], action[1][0])
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
        if rot_angle < 0:
            rot_angle = 360 + rot_angle

        rotate_idx = round(rot_angle / (360 / NUM_ROTATION))
        if rotate_idx == NUM_ROTATION:
            rotate_idx = 0

        new_push_predictions[
            rotate_idx,
            action_start[1] - 3 : action_start[1] + 4,
            action_start[0] - 3 : action_start[0] + 4,
        ] = push_predictions[
            rotate_idx,
            action_start[1] - 3 : action_start[1] + 4,
            action_start[0] - 3 : action_start[0] + 4,
        ]
        print(
            np.max(
                push_predictions[
                    rotate_idx,
                    action_start[1] - 3 : action_start[1] + 4,
                    action_start[0] - 3 : action_start[0] + 4,
                ]
            )
        )
        # new_push_predictions[rotate_idx, action_start[1], action_start[0]] = push_predictions[rotate_idx, action_start[1], action_start[0]]

        # best_locate = [rot_angle, action_start[1], action_start[0], action_end[1], action_end[0]]
        # action_start = (best_locate[1], best_locate[2])

        # rotated_color_image = utils.rotate(color_image, rot_angle)
        # origin = mask_image.shape
        # origin = ((origin[0] - 1) / 2, (origin[1] - 1) / 2)
        # new_action_start = utils.rotate_point(origin, action_start, math.radians(rot_angle))
        # new_action_start = (round(new_action_start[0]), round(new_action_start[1]))
        # point_from = (int(new_action_start[1]), int(new_action_start[0]))
        # point_to = (int(point_from[0] + PUSH_DISTANCE_PIXEL), int(point_from[1]))
        # rotated_color_image = cv2.arrowedLine(
        #     rotated_color_image, point_from, point_to, (100, 200, 0), 2, tipLength=0.2,
        # )
        # cv2.imshow('before', color_image)
        # cv2.imshow('after', rotated_color_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    return new_push_predictions


@torch.no_grad()
def get_q_mask(model, mask_heightmap, env):
    mask_heightmap = np.copy(mask_heightmap)

    # relabel
    mask_heightmap = utils.relabel_mask(env, mask_heightmap)

    # focus on target, so make one extra channel
    target_mask_img = np.zeros_like(mask_heightmap, dtype=np.uint8)
    target_mask_img[mask_heightmap == 255] = 255
    mask_heightmap = np.dstack((target_mask_img, mask_heightmap))

    # Add extra padding (to handle rotations inside network)
    mask_heightmap = np.pad(
        mask_heightmap,
        ((IMAGE_PAD_WIDTH, IMAGE_PAD_WIDTH), (IMAGE_PAD_WIDTH, IMAGE_PAD_WIDTH), (0, 0)),
        "constant",
        constant_values=0,
    )

    input_image = mask_heightmap.astype(float) / 255

    # Construct minibatch of size 1 (b,c,h,w)
    input_image.shape = (
        input_image.shape[0],
        input_image.shape[1],
        input_image.shape[2],
        1,
    )
    input_data = torch.from_numpy(input_image.astype(np.float32)).permute(3, 2, 0, 1)

    # Pass input data through model
    # output_prob = model(input_color_data, input_depth_data, True, -1, use_push=True, push_only=True)
    output_prob = model(input_data, True, -1)

    # Return Q values (and remove extra padding)
    for rotate_idx in range(len(output_prob)):
        # output = torch.sigmoid(output_prob[rotate_idx])
        output = output_prob[rotate_idx]
        if rotate_idx == 0:
            push_predictions = output.cpu().data.numpy()[
                :, 0, :, :,
            ]
        else:
            push_predictions = np.concatenate(
                (push_predictions, output.cpu().data.numpy()[:, 0, :, :,],), axis=0,
            )
    padding_width_start = IMAGE_PAD_WIDTH
    padding_width_end = push_predictions[0].shape[0] - IMAGE_PAD_WIDTH
    push_predictions = push_predictions[
        :, padding_width_start:padding_width_end, padding_width_start:padding_width_end
    ]

    best_pix_ind = np.unravel_index(np.argmax(push_predictions), push_predictions.shape)
    grasp_q_value = push_predictions[best_pix_ind]

    return grasp_q_value, best_pix_ind, push_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Lifelong DQN")

    parser.add_argument("--test_case", action="store", help="File for testing")

    parser.add_argument("--test_cases", nargs="+", help="Files for testing")

    parser.add_argument(
        "--max_test_trials",
        action="store",
        type=int,
        default=5,
        help="maximum number of test runs per case/scenario",
    )

    parser.add_argument(
        "--num_iter",
        action="store",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--push_model",
        action="store",
        type=str,
        default="logs_mcts/runs/2021-09-02-22-59-train-ratio-1-final/lifelong_model-20.pth",
    )

    parser.add_argument("--switch", action="store", type=int, help="Switch target")

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    # set seed
    random.seed(1234)
    torch.manual_seed(1234)
    np.random.seed(1234)

    # network only?
    iteration = 0
    args = parse_args()
    case = args.test_case
    cases = args.test_cases
    switch = args.switch
    if switch is not None:
        print(f"Target ID has been switched to {switch}")
    if cases:
        repeat_num = len(cases)
    else:
        repeat_num = args.max_test_trials
        cases = [case] * repeat_num
    collector = SeachCollector(cases)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    env = Environment(gui=False)
    mcts_helper = MCTSHelper(env, "logs_grasp/snapshot-post-020000.reinforcement.pth")
    push_model = PushNet()
    push_model.load_state_dict(torch.load(args.push_model)["model"])
    push_model = push_model.to(device)
    push_model.eval()

    num_action_log = defaultdict(list)
    for repeat_idx in range(repeat_num):
        success = False
        while not success:
            env.reset()
            success = env.add_object_push_from_file(cases[repeat_idx])
            print(f"Reset environment of {repeat_idx}")

        num_action = [0, 0, 0]
        start_time = time.time()
        while True:
            num_action[0] += 1
            color_image, depth_image, _ = utils.get_true_heightmap(env)
            temp = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(temp, TARGET_LOWER, TARGET_UPPER)
            print(f"Target on the table (value: {np.sum(mask) / 255})")
            if np.sum(mask) / 255 < 10:
                break
            q_value, best_pix_ind, grasp_predictions = mcts_helper.get_grasp_q(
                color_image, depth_image, post_checking=True
            )
            print(f"Max grasp Q value: {q_value}")

            # record
            collector.save_heightmaps(iteration, color_image, depth_image)
            grasp_pred_vis = mcts_helper.get_prediction_vis(
                grasp_predictions, color_image, best_pix_ind
            )
            collector.save_visualizations(iteration, grasp_pred_vis, "grasp")

            # Grasp >>>>>
            if q_value > GRASP_Q_GRASP_THRESHOLD:
                best_rotation_angle = np.deg2rad(best_pix_ind[0] * (360.0 / NUM_ROTATION))
                primitive_position = [
                    best_pix_ind[1] * PIXEL_SIZE + WORKSPACE_LIMITS[0][0],
                    best_pix_ind[2] * PIXEL_SIZE + WORKSPACE_LIMITS[1][0],
                    depth_image[best_pix_ind[1]][best_pix_ind[2]] + WORKSPACE_LIMITS[2][0],
                ]
                if not IS_REAL:
                    success = env.grasp(primitive_position, best_rotation_angle)
                else:
                    grasp_sucess = env.grasp(primitive_position, best_rotation_angle)
                    success = grasp_sucess
                # record
                reward_value = 1 if success else 0
                collector.executed_action_log.append(
                    [
                        1,  # grasp
                        primitive_position[0],
                        primitive_position[1],
                        primitive_position[2],
                        best_rotation_angle,
                        -1,
                        -1,
                    ]
                )
                collector.label_value_log.append(reward_value)
                collector.write_to_log("executed-action", collector.executed_action_log)
                collector.write_to_log("label-value", collector.label_value_log)
                iteration += 1
                if success:
                    num_action[2] += 1
                    break
                else:
                    continue
            # Grasp <<<<<

            # Push >>>>>
            num_action[1] += 1
            color_image, depth_image, mask_image = utils.get_true_heightmap(env)
            start, end = sampled_prediction_precise(
                mcts_helper, env, push_model, color_image, mask_image
            )
            primitive_position = [
                start[0] * PIXEL_SIZE + WORKSPACE_LIMITS[0][0],
                start[1] * PIXEL_SIZE + WORKSPACE_LIMITS[1][0],
                0.01,
            ]
            primitive_position_end = [
                end[0] * PIXEL_SIZE + WORKSPACE_LIMITS[0][0],
                end[1] * PIXEL_SIZE + WORKSPACE_LIMITS[1][0],
                0.01,
            ]
            env.push(primitive_position, primitive_position_end)
            # Push <<<<<
            # Push >>>>>
            # num_action[1] += 1
            # color_image, depth_image, mask_image = utils.get_true_heightmap(env)
            # q_value, best_pix_ind, predictions = get_q_mask(push_model, mask_image, env)
            # use same action space as mcts >>>>>
            # predictions = sampled_prediction(mcts_helper, env, color_image, mask_image, predictions)
            # best_pix_ind = np.unravel_index(np.argmax(predictions), predictions.shape)
            # grasp_q_value = predictions[best_pix_ind]
            # # <<<<<
            # print(f"Push {q_value}")
            # best_rotation_angle = np.deg2rad(90 - best_pix_ind[0] * (360.0 / NUM_ROTATION))
            # primitive_position = [
            #     best_pix_ind[1] * PIXEL_SIZE + WORKSPACE_LIMITS[0][0],
            #     best_pix_ind[2] * PIXEL_SIZE + WORKSPACE_LIMITS[1][0],
            #     0.01,
            # ]
            # primitive_position_end = [
            #     primitive_position[0] + PUSH_DISTANCE * np.cos(best_rotation_angle),
            #     primitive_position[1] + PUSH_DISTANCE * np.sin(best_rotation_angle),
            #     0.01,
            # ]
            # env.push(primitive_position, primitive_position_end)
            # pred_vis = mcts_helper.get_prediction_vis(
            #     predictions, color_image, best_pix_ind, is_push=True
            # )
            # cv2.imwrite(
            #     "vis.png", pred_vis,
            # )
            # input("wait")
            # record
            reward_value = 0
            collector.executed_action_log.append(
                [
                    0,  # push
                    primitive_position[0],
                    primitive_position[1],
                    primitive_position[2],
                    primitive_position_end[0],
                    primitive_position_end[1],
                    primitive_position_end[2],
                ]
            )
            collector.label_value_log.append(reward_value)
            collector.write_to_log("executed-action", collector.executed_action_log)
            collector.write_to_log("label-value", collector.label_value_log)
            iteration += 1
            # Push <<<<<

        end_time = time.time()
        collector.time_log.append(end_time - start_time)
        collector.write_to_log("executed-time", collector.time_log)

        print(num_action)
        num_action_log[cases[repeat_idx]].append(num_action)



    print(num_action_log)
    total_case = 0
    total_action = 0
    for key in num_action_log:
        this_total_case = 0
        this_total_action = 0
        this_total_push = 0
        this_total_grasp = 0
        this_total_success = 0
        for trial in num_action_log[key]:
            this_total_case += 1
            this_total_action += trial[0]
            this_total_push += trial[1]
            this_total_grasp += (trial[0] - trial[1])
            this_total_success += trial[2]
        print(key, "this_case:", this_total_case, "this_action:", this_total_action,
            "this_push:", this_total_push, "this_grasp:", this_total_grasp,
            "average num", this_total_action/this_total_case,  "average_grasp", this_total_success / this_total_grasp, "total_complete", this_total_success
        )
        total_case += len(num_action_log[key])
        for re in num_action_log[key]:
            total_action += re[0]
    print(total_case, total_action, total_action / total_case)
