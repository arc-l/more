from dataset import LifelongEvalDataset
import math
import random

import torch
from torchvision.transforms import functional as TF
import numpy as np
import cv2
import imutils

from models import reinforcement_net
from action_utils_mask import get_orientation, adjust_push_start_point
import utils
from constants import (
    GRIPPER_PUSH_RADIUS_PIXEL,
    GRIPPER_PUSH_RADIUS_SAFE_PAD_PIXEL,
    PIXEL_SIZE,
    PUSH_DISTANCE_PIXEL,
    TARGET_LOWER,
    TARGET_UPPER,
    IMAGE_PAD_WIDTH,
    COLOR_MEAN,
    COLOR_STD,
    DEPTH_MEAN,
    DEPTH_STD,
    NUM_ROTATION,
    GRIPPER_GRASP_INNER_DISTANCE_PIXEL,
    GRIPPER_GRASP_WIDTH_PIXEL,
    GRIPPER_GRASP_SAFE_WIDTH_PIXEL,
    GRIPPER_GRASP_OUTER_DISTANCE_PIXEL,
    IMAGE_PAD_WIDTH,
    BG_THRESHOLD,
    IMAGE_SIZE,
    WORKSPACE_LIMITS,
    PUSH_DISTANCE,
)


class MCTSHelper:
    """
    Simulate the state after push actions.
    Evaluation the grasp rewards.
    """

    def __init__(self, env, grasp_model_path):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Initialize Mask R-CNN
        # self.mask_model = get_model_instance_segmentation(2)
        # self.mask_model.load_state_dict(torch.load(mask_model_path))
        # self.mask_model = self.mask_model.to(self.device)
        # self.mask_model.eval()

        # Initialize Grasp Q Evaluation
        self.grasp_model = reinforcement_net()
        self.grasp_model.load_state_dict(torch.load(grasp_model_path)["model"], strict=False)
        self.grasp_model = self.grasp_model.to(self.device)
        self.grasp_model.eval()

        self.env = env
        self.move_recorder = {}
        self.simulation_recorder = {}

    def reset(self):
        self.move_recorder = {}
        self.simulation_recorder = {}

    # @torch.no_grad()
    # def from_maskrcnn(self, color_image, plot=False):
    #     """
    #     Use Mask R-CNN to do instance segmentation and output masks in binary format.
    #     """
    #     image = color_image.copy()
    #     image = TF.to_tensor(image)
    #     prediction = self.mask_model([image.to(self.device)])[0]

    #     mask_objs = []
    #     if plot:
    #         pred_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    #     for idx, mask in enumerate(prediction["masks"]):
    #         # NOTE: 0.98 can be tuned
    #         if prediction["scores"][idx] > 0.98:
    #             img = mask[0].mul(255).byte().cpu().numpy()
    #             img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    #             if np.sum(img == 255) < 100:
    #                 continue
    #             mask_objs.append(img)
    #             if plot:
    #                 pred_mask[img > 0] = 255 - idx * 50
    #                 cv2.imwrite(str(idx) + "mask.png", img)
    #     if plot:
    #         cv2.imwrite("pred.png", pred_mask)
    #     print("Mask R-CNN: %d objects detected" % len(mask_objs), prediction["scores"].cpu())
    #     return mask_objs

    # def sample_actions(
    #     self, object_states, color_image=None, mask_image=None, prev_move=None, plot=False
    # ):
    #     """
    #     Sample actions around the objects, from the boundary to the center.
    #     Assume there is no object in "black"
    #     Output the rotated image, such that the push action is from left to right
    #     """

    #     # Retrieve information
    #     if color_image is None:
    #         self.env.restore_objects(object_states)
    #         color_image, _, mask_image = utils.get_true_heightmap(self.env)

    #     # Process mask into binary format
    #     masks = []
    #     for i in self.env.obj_ids["rigid"]:
    #         mask = np.where(mask_image == i, 255, 0).astype(np.uint8)
    #         masks.append(mask)
    #     if len(masks) == 0:
    #         return [], [], [], [], [], [], [], []

    #     gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
    #     gray = gray.astype(np.uint8)
    #     if plot:
    #         plot_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    #     blurred = cv2.medianBlur(gray, 5)
    #     thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)[1]
    #     cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #     cnts = imutils.grab_contours(cnts)

    #     # find the contour of a single object
    #     points_on_contour = []
    #     points = []
    #     # four_idx = []
    #     other_idx = []
    #     # priority_points_on_contour = []
    #     # priority_points = []
    #     for oi in range(len(masks)):
    #         obj_cnt = cv2.findContours(masks[oi], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #         obj_cnt = imutils.grab_contours(obj_cnt)
    #         obj_cnt = sorted(obj_cnt, key=lambda x: cv2.contourArea(x))
    #         if len(obj_cnt) == 0:
    #             continue
    #         else:
    #             obj_cnt = obj_cnt[-1]
    #             # if too small, then, we skip
    #             if cv2.contourArea(obj_cnt) < 10:
    #                 continue
    #         # get center
    #         M = cv2.moments(obj_cnt)
    #         cX = round(M["m10"] / M["m00"])
    #         cY = round(M["m01"] / M["m00"])
    #         if plot:
    #             cv2.circle(plot_image, (cX, cY), 3, (255, 255, 255), -1)
    #         # get pca angle
    #         # angle = get_orientation(obj_cnt)
    #         # get contour points
    #         skip_num = len(obj_cnt) // 12  # 12 possible pushes for an object
    #         skip_count = 0
    #         # diff_angle_limit_four = 0.3
    #         # target_diff_angles = np.array([0, np.pi, np.pi / 2, 3 * np.pi / 2])
    #         # add the consecutive move
    #         # if prev_move:
    #         #     prev_angle = math.atan2(
    #         #         prev_move[1][1] - prev_move[0][1], prev_move[1][0] - prev_move[0][0]
    #         #     )
    #         #     pose = (cX - math.cos(prev_angle) * 2, cY - math.sin(prev_angle) * 2)
    #         #     x = pose[0]
    #         #     y = pose[1]
    #         #     diff_x = cX - x
    #         #     diff_y = cY - y
    #         #     diff_norm = math.sqrt(diff_x ** 2 + diff_y ** 2)
    #         #     diff_x /= diff_norm
    #         #     diff_y /= diff_norm
    #         #     point_on_contour = (round(x), round(y))
    #         #     diff_mul = adjust_push_start_point((cX, cY), point_on_contour, obj_cnt)
    #         #     point = (round(x - diff_x * diff_mul), round(y - diff_y * diff_mul))
    #         #     diff_mul = adjust_push_start_point(
    #         #         (cX, cY), point_on_contour, obj_cnt, add_distance=0
    #         #     )
    #         #     test_point = (round(x - diff_x * diff_mul), round(y - diff_y * diff_mul))
    #         #     if is_close(prev_move[1], test_point):
    #         #         if len(priority_points) > 0:
    #         #             prev_dis = close_distance(prev_move[1], priority_points[0])
    #         #             this_dis = close_distance(prev_move[1], test_point)
    #         #             if this_dis < prev_dis:
    #         #                 priority_points_on_contour[0] = point_on_contour
    #         #                 priority_points[0] = point
    #         #         else:
    #         #             priority_points_on_contour.append(point_on_contour)
    #         #             priority_points.append(point)
    #         # add four directions to center of object
    #         # four_poses = [
    #         #     (cX + math.cos(angle) * 2, cY + math.sin(angle) * 2),
    #         #     (cX + math.cos(angle + np.pi / 2) * 2, cY + math.sin(angle + np.pi / 2) * 2),
    #         #     (cX + math.cos(angle - np.pi / 2) * 2, cY + math.sin(angle - np.pi / 2) * 2),
    #         #     (cX - math.cos(angle) * 2, cY - math.sin(angle) * 2),
    #         # ]
    #         # for pose in four_poses:
    #         #     x = pose[0]
    #         #     y = pose[1]
    #         #     diff_x = cX - x
    #         #     diff_y = cY - y
    #         #     diff_norm = math.sqrt(diff_x ** 2 + diff_y ** 2)
    #         #     diff_x /= diff_norm
    #         #     diff_y /= diff_norm
    #         #     point_on_contour = (round(x), round(y))
    #         #     diff_mul = adjust_push_start_point((cX, cY), point_on_contour, obj_cnt)
    #         #     point = (round(x - diff_x * diff_mul), round(y - diff_y * diff_mul))
    #         #     points_on_contour.append(point_on_contour)
    #         #     points.append(point)
    #         #     four_idx.append(len(points) - 1)
    #         tested_angles = []
    #         for pi, p in enumerate(obj_cnt):
    #             x = p[0][0]
    #             y = p[0][1]
    #             if x == cX or y == cY:
    #                 continue
    #             diff_x = cX - x
    #             diff_y = cY - y
    #             test_angle = math.atan2(diff_y, diff_x)
    #             should_append = False
    #             # avoid four directions to center of object
    #             # if (
    #             #     np.min(np.abs(abs(angle - test_angle) - target_diff_angles))
    #             #     < diff_angle_limit_four
    #             # ):
    #             #     should_append = False
    #             #     skip_count = 0
    #             if skip_count == skip_num:
    #                 should_append = True
    #             if should_append:
    #                 diff_norm = math.sqrt(diff_x ** 2 + diff_y ** 2)
    #                 diff_x /= diff_norm
    #                 diff_y /= diff_norm
    #                 point_on_contour = (round(x), round(y))
    #                 diff_mul = adjust_push_start_point((cX, cY), point_on_contour, obj_cnt)
    #                 point = (round(x - diff_x * diff_mul), round(y - diff_y * diff_mul))
    #                 points_on_contour.append(point_on_contour)
    #                 points.append(point)
    #                 other_idx.append(len(points) - 1)
    #                 skip_count = 0
    #                 tested_angles.append(test_angle)
    #             else:
    #                 skip_count += 1
    #     # random actions, adding priority points at the end
    #     # random.shuffle(four_idx)
    #     random.shuffle(other_idx)
    #     new_points = []
    #     new_points_on_contour = []
    #     for idx in other_idx:
    #         new_points.append(points[idx])
    #         new_points_on_contour.append(points_on_contour[idx])
    #     # for idx in four_idx:
    #     #     new_points.append(points[idx])
    #     #     new_points_on_contour.append(points_on_contour[idx])
    #     # new_points.extend(priority_points)
    #     # new_points_on_contour.extend(priority_points_on_contour)
    #     points = new_points
    #     points_on_contour = new_points_on_contour

    #     if plot:
    #         # loop over the contours
    #         for c in cnts:
    #             cv2.drawContours(plot_image, [c], -1, (133, 137, 140), 2)

    #     actions = []
    #     for pi in range(len(points)):
    #         # out of boundary
    #         if (
    #             points[pi][0] < 5
    #             or points[pi][0] > IMAGE_SIZE - 5
    #             or points[pi][1] < 5
    #             or points[pi][1] > IMAGE_SIZE - 5
    #         ):
    #             qualify = False
    #         # consecutive action
    #         # elif pi >= len(points) - len(priority_points):
    #         #     qualify = True
    #         # clearance large
    #         elif (
    #             np.sum(
    #                 thresh[
    #                     max(0, points[pi][1] - GRIPPER_PUSH_RADIUS_SAFE_PIXEL) : min(
    #                         IMAGE_SIZE, points[pi][1] + GRIPPER_PUSH_RADIUS_SAFE_PIXEL + 1
    #                     ),
    #                     max(0, points[pi][0] - GRIPPER_PUSH_RADIUS_SAFE_PIXEL) : min(
    #                         IMAGE_SIZE, points[pi][0] + GRIPPER_PUSH_RADIUS_SAFE_PIXEL + 1
    #                     ),
    #                 ]
    #                 > 0
    #             )
    #             == 0
    #         ):
    #             qualify = True
    #         # clearance small
    #         else:
    #             # compute rotation angle
    #             down = (0, 1)
    #             current = (
    #                 points_on_contour[pi][0] - points[pi][0],
    #                 points_on_contour[pi][1] - points[pi][1],
    #             )
    #             dot = (
    #                 down[0] * current[0] + down[1] * current[1]
    #             )  # dot product between [x1, y1] and [x2, y2]
    #             det = down[0] * current[1] - down[1] * current[0]  # determinant
    #             angle = math.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
    #             angle = math.degrees(angle)
    #             crop = thresh[
    #                 points[pi][1]
    #                 - GRIPPER_PUSH_RADIUS_SAFE_PIXEL : points[pi][1]
    #                 + GRIPPER_PUSH_RADIUS_SAFE_PIXEL
    #                 + 1,
    #                 points[pi][0]
    #                 - GRIPPER_PUSH_RADIUS_SAFE_PIXEL : points[pi][0]
    #                 + GRIPPER_PUSH_RADIUS_SAFE_PIXEL
    #                 + 1,
    #             ]
    #             if crop.shape == (
    #                 GRIPPER_PUSH_RADIUS_SAFE_PIXEL * 2 + 1,
    #                 GRIPPER_PUSH_RADIUS_SAFE_PIXEL * 2 + 1,
    #             ):
    #                 crop = utils.rotate(crop, angle)
    #                 (h, w) = crop.shape
    #                 crop_cy, crop_cx = (h // 2, w // 2)
    #                 crop = crop[
    #                     crop_cy
    #                     - math.ceil(GRIPPER_GRASP_WIDTH_PIXEL / 2) : crop_cy
    #                     + math.ceil(GRIPPER_GRASP_WIDTH_PIXEL / 2)
    #                     + 1,
    #                     crop_cx
    #                     - GRIPPER_PUSH_RADIUS_PIXEL : crop_cx
    #                     + GRIPPER_PUSH_RADIUS_PIXEL
    #                     + 1,
    #                 ]
    #                 qualify = np.sum(crop > 0) == 0
    #             else:
    #                 qualify = False
    #         if qualify:
    #             if plot:
    #                 diff_x = points_on_contour[pi][0] - points[pi][0]
    #                 diff_y = points_on_contour[pi][1] - points[pi][1]
    #                 diff_norm = math.sqrt(diff_x ** 2 + diff_y ** 2)
    #                 diff_x /= diff_norm
    #                 diff_y /= diff_norm
    #                 point_to = (
    #                     int(points[pi][0] + diff_x * PUSH_DISTANCE / PIXEL_SIZE / 2),
    #                     int(points[pi][1] + diff_y * PUSH_DISTANCE / PIXEL_SIZE / 2),
    #                 )
    #                 if pi < len(other_idx):
    #                     cv2.arrowedLine(
    #                         plot_image, points[pi], point_to, (0, 0, 255), 2, tipLength=0.2,
    #                     )
    #                 # elif pi >= len(points) - len(priority_points):
    #                 #     cv2.arrowedLine(
    #                 #         plot_image, tuple(points[pi]), point_to, (0, 255, 0), 2, tipLength=0.2,
    #                 #     )
    #                 else:
    #                     cv2.arrowedLine(
    #                         plot_image, points[pi], point_to, (255, 0, 0), 2, tipLength=0.2,
    #                     )
    #             push_start = (points[pi][1], points[pi][0])
    #             push_vector = np.array(
    #                 [
    #                     points_on_contour[pi][1] - points[pi][1],
    #                     points_on_contour[pi][0] - points[pi][0],
    #                 ]
    #             )
    #             unit_push = push_vector / np.linalg.norm(push_vector)
    #             push_end = (
    #                 round(push_start[0] + unit_push[0] * PUSH_DISTANCE / PIXEL_SIZE),
    #                 round(push_start[1] + unit_push[1] * PUSH_DISTANCE / PIXEL_SIZE),
    #             )
    #             actions.append([push_start, push_end])

    #     if plot:
    #         cv2.imwrite("test.png", plot_image)

    #     return actions

    def check_valid(self, point, point_on_contour, thresh):
        # out of boundary
        if not (
            GRIPPER_PUSH_RADIUS_SAFE_PAD_PIXEL
            < point[0]
            < IMAGE_SIZE - GRIPPER_PUSH_RADIUS_SAFE_PAD_PIXEL
        ) or not (
            GRIPPER_PUSH_RADIUS_SAFE_PAD_PIXEL
            < point[1]
            < IMAGE_SIZE - GRIPPER_PUSH_RADIUS_SAFE_PAD_PIXEL
        ):
            qualify = False
        else:
            # compute rotation angle
            down = (0, 1)
            current = (
                point_on_contour[0] - point[0],
                point_on_contour[1] - point[1],
            )
            dot = (
                down[0] * current[0] + down[1] * current[1]
            )  # dot product between [x1, y1] and [x2, y2]
            det = down[0] * current[1] - down[1] * current[0]  # determinant
            angle = math.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
            angle = math.degrees(angle)
            crop = thresh[
                point[1]
                - GRIPPER_PUSH_RADIUS_SAFE_PAD_PIXEL : point[1]
                + GRIPPER_PUSH_RADIUS_SAFE_PAD_PIXEL
                + 1,
                point[0]
                - GRIPPER_PUSH_RADIUS_SAFE_PAD_PIXEL : point[0]
                + GRIPPER_PUSH_RADIUS_SAFE_PAD_PIXEL
                + 1,
            ]
            # test the rotated crop part
            crop = utils.rotate(crop, angle, is_mask=True)
            (h, w) = crop.shape
            crop_cy, crop_cx = (h // 2, w // 2)
            crop = crop[
                crop_cy
                - math.ceil(GRIPPER_GRASP_WIDTH_PIXEL / 2)
                - 1 : crop_cy
                + math.ceil(GRIPPER_GRASP_WIDTH_PIXEL / 2)
                + 2,
                crop_cx - GRIPPER_PUSH_RADIUS_PIXEL - 1 : crop_cx + GRIPPER_PUSH_RADIUS_PIXEL + 2,
            ]
            qualify = np.sum(crop > 0) == 0

        return qualify

    def global_adjust(self, point, point_on_contour, thresh):
        for dis in [0.01, 0.02]:
            dis = dis / PIXEL_SIZE
            diff_x = point_on_contour[0] - point[0]
            diff_y = point_on_contour[1] - point[1]
            diff_norm = math.sqrt(diff_x ** 2 + diff_y ** 2)
            diff_x /= diff_norm
            diff_y /= diff_norm
            test_point = (round(point[0] - diff_x * dis), round(point[1] - diff_y * dis))
            qualify = self.check_valid(test_point, point_on_contour, thresh)
            if qualify:
                return qualify, test_point

        return False, None

    def sample_actions(
        self, object_states, color_image=None, mask_image=None, env=None, plot=False, masks=None
    ):
        """
        Sample actions around the objects, from the boundary to the center.
        Assume there is no object in "black"
        Output the rotated image, such that the push action is from left to right
        """

        if env is None:
            env = self.env

        # Retrieve information
        if color_image is None:
            env.restore_objects(object_states)
            color_image, _, mask_image = utils.get_true_heightmap(env)

        # Process mask into binary format
        if masks is None:
            masks = []
            for i in env.obj_ids["rigid"]:
                mask = np.where(mask_image == i, 255, 0).astype(np.uint8)
                masks.append(mask)
            if len(masks) == 0:
                return None

        gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
        gray = gray.astype(np.uint8)
        if plot:
            plot_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        blurred = cv2.medianBlur(gray, 5)
        thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnts = imutils.grab_contours(cnts)

        # find the contour of a single object
        points_on_contour = []
        points = []
        four_idx = []
        other_idx = []
        for oi in range(len(masks)):
            obj_cnt = cv2.findContours(masks[oi], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            obj_cnt = imutils.grab_contours(obj_cnt)
            obj_cnt = sorted(obj_cnt, key=lambda x: cv2.contourArea(x))
            if len(obj_cnt) == 0:
                continue
            else:
                obj_cnt = obj_cnt[-1]
                # if too small, then, we skip
                if cv2.contourArea(obj_cnt) < 10:
                    continue
            # get center
            M = cv2.moments(obj_cnt)
            cX = round(M["m10"] / M["m00"])
            cY = round(M["m01"] / M["m00"])
            if plot:
                cv2.circle(plot_image, (cX, cY), 3, (255, 255, 255), -1)
            # get pca angle
            angle = get_orientation(obj_cnt)
            # get contour points
            # skip_num = len(obj_cnt) // 12  # 12 possible pushes for an object
            # skip_count = 0
            diff_angle_limit = 0.75  # around 45 degrees
            # target_diff_angles = np.array([0, np.pi, np.pi / 2, 3 * np.pi / 2])
            target_diff_angles = []
            # add four directions to center of object
            four_poses = [
                (cX + math.cos(angle) * 2, cY + math.sin(angle) * 2),
                (cX + math.cos(angle + np.pi / 2) * 2, cY + math.sin(angle + np.pi / 2) * 2),
                (cX + math.cos(angle - np.pi / 2) * 2, cY + math.sin(angle - np.pi / 2) * 2),
                (cX - math.cos(angle) * 2, cY - math.sin(angle) * 2),
            ]
            for pose in four_poses:
                x = pose[0]
                y = pose[1]
                diff_x = cX - x
                diff_y = cY - y
                test_angle = math.atan2(diff_y, diff_x)
                diff_norm = math.sqrt(diff_x ** 2 + diff_y ** 2)
                diff_x /= diff_norm
                diff_y /= diff_norm
                point_on_contour = (round(x), round(y))
                diff_mul = adjust_push_start_point((cX, cY), point_on_contour, obj_cnt)
                point = (round(x - diff_x * diff_mul), round(y - diff_y * diff_mul))
                should_append = self.check_valid(point, point_on_contour, thresh)
                if not should_append:
                    should_append, point = self.global_adjust(point, point_on_contour, thresh)
                if should_append:
                    points_on_contour.append(point_on_contour)
                    points.append(point)
                    four_idx.append(len(points) - 1)
                    target_diff_angles.append(test_angle)
            for pi, p in enumerate(obj_cnt):
                x = p[0][0]
                y = p[0][1]
                if x == cX or y == cY:
                    continue
                diff_x = cX - x
                diff_y = cY - y
                test_angle = math.atan2(diff_y, diff_x)
                # avoid similar directions to center of object
                if len(target_diff_angles) > 0:
                    test_target_diff_angles = np.abs(np.array(target_diff_angles) - test_angle)
                    should_append = (
                        np.min(test_target_diff_angles) > diff_angle_limit
                        and np.max(test_target_diff_angles) < math.pi * 2 - diff_angle_limit
                    )
                else:
                    should_append = True
                if should_append:
                    diff_norm = math.sqrt(diff_x ** 2 + diff_y ** 2)
                    diff_x /= diff_norm
                    diff_y /= diff_norm
                    point_on_contour = (round(x), round(y))
                    diff_mul = adjust_push_start_point((cX, cY), point_on_contour, obj_cnt)
                    point = (round(x - diff_x * diff_mul), round(y - diff_y * diff_mul))
                    should_append = self.check_valid(point, point_on_contour, thresh)
                    if not should_append:
                        should_append, point = self.global_adjust(point, point_on_contour, thresh)
                    if should_append:
                        points_on_contour.append(point_on_contour)
                        points.append(point)
                        other_idx.append(len(points) - 1)
                        target_diff_angles.append(test_angle)
        # random actions, adding priority points at the end
        random.shuffle(four_idx)
        random.shuffle(other_idx)
        new_points = []
        new_points_on_contour = []
        for idx in other_idx:
            new_points.append(points[idx])
            new_points_on_contour.append(points_on_contour[idx])
        for idx in four_idx:
            new_points.append(points[idx])
            new_points_on_contour.append(points_on_contour[idx])
        points = new_points
        points_on_contour = new_points_on_contour
        idx_list = list(range(len(points)))
        random.shuffle(idx_list)
        new_points = []
        new_points_on_contour = []
        for idx in idx_list:
            new_points.append(points[idx])
            new_points_on_contour.append(points_on_contour[idx])
        points = new_points
        points_on_contour = new_points_on_contour

        if plot:
            # loop over the contours
            for c in cnts:
                cv2.drawContours(plot_image, [c], -1, (133, 137, 140), 2)

        actions = []
        for pi in range(len(points)):
            if plot:
                diff_x = points_on_contour[pi][0] - points[pi][0]
                diff_y = points_on_contour[pi][1] - points[pi][1]
                diff_norm = math.sqrt(diff_x ** 2 + diff_y ** 2)
                diff_x /= diff_norm
                diff_y /= diff_norm
                point_to = (
                    int(points[pi][0] + diff_x * PUSH_DISTANCE / PIXEL_SIZE / 2),
                    int(points[pi][1] + diff_y * PUSH_DISTANCE / PIXEL_SIZE / 2),
                )
                if pi < len(other_idx):
                    cv2.arrowedLine(
                        plot_image, points[pi], point_to, (0, 0, 255), 2, tipLength=0.2,
                    )
                else:
                    cv2.arrowedLine(
                        plot_image, points[pi], point_to, (255, 0, 0), 2, tipLength=0.2,
                    )
            push_start = (points[pi][1], points[pi][0])
            push_vector = np.array(
                [
                    points_on_contour[pi][1] - points[pi][1],
                    points_on_contour[pi][0] - points[pi][0],
                ]
            )
            unit_push = push_vector / np.linalg.norm(push_vector)
            push_end = (
                round(push_start[0] + unit_push[0] * PUSH_DISTANCE / PIXEL_SIZE),
                round(push_start[1] + unit_push[1] * PUSH_DISTANCE / PIXEL_SIZE),
            )
            actions.append([push_start, push_end])

        if plot:
            cv2.imwrite("test.png", plot_image)

        return actions

    def simulate(self, push_start, push_end, restore_states=None):
        if restore_states is not None:
            self.env.restore_objects(restore_states)
        push_start = [
            push_start[0] * PIXEL_SIZE + WORKSPACE_LIMITS[0][0],
            push_start[1] * PIXEL_SIZE + WORKSPACE_LIMITS[1][0],
            0.01,
        ]
        push_end = [
            push_end[0] * PIXEL_SIZE + WORKSPACE_LIMITS[0][0],
            push_end[1] * PIXEL_SIZE + WORKSPACE_LIMITS[1][0],
            0.01,
        ]

        success = self.env.push(push_start, push_end, verbose=False)
        if not success:
            return None
        self.env.wait_static()
        object_states = self.env.save_objects()

        # Check if all objects are still in workspace
        for obj in object_states:
            pos = obj[0]
            if (
                pos[0] < WORKSPACE_LIMITS[0][0]
                or pos[0] > WORKSPACE_LIMITS[0][1]
                or pos[1] < WORKSPACE_LIMITS[1][0]
                or pos[1] > WORKSPACE_LIMITS[1][1]
            ):
                return None

        color_image, depth_image, mask_image = utils.get_true_heightmap(self.env)

        return color_image, depth_image, mask_image, object_states

    @torch.no_grad()
    def get_grasp_q(self, color_heightmap, depth_heightmap, post_checking=False, is_real=False):
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
        input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(
            3, 2, 0, 1
        )
        input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(
            3, 2, 0, 1
        )

        # Pass input data through model
        output_prob = self.grasp_model(input_color_data, input_depth_data, True, -1, False)

        # Return Q values (and remove extra padding)
        for rotate_idx in range(len(output_prob)):
            if rotate_idx == 0:
                grasp_predictions = (
                    output_prob[rotate_idx][1].cpu().data.numpy()[:, 0, :, :,]
                )
            else:
                grasp_predictions = np.concatenate(
                    (
                        grasp_predictions,
                        output_prob[rotate_idx][1].cpu().data.numpy()[:, 0, :, :,],
                    ),
                    axis=0,
                )

        # post process, only grasp one object, focus on blue object
        temp = cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(temp, TARGET_LOWER, TARGET_UPPER)
        mask_pad = np.pad(mask, IMAGE_PAD_WIDTH, "constant", constant_values=0)
        mask_bg = cv2.inRange(temp, BG_THRESHOLD["low"], BG_THRESHOLD["high"])
        mask_bg_pad = np.pad(mask_bg, IMAGE_PAD_WIDTH, "constant", constant_values=255)
        # focus on blue
        for rotate_idx in range(len(grasp_predictions)):
            grasp_predictions[rotate_idx][mask_pad != 255] = 0
        padding_width_start = IMAGE_PAD_WIDTH
        padding_width_end = grasp_predictions[0].shape[0] - IMAGE_PAD_WIDTH
        # only grasp one object
        kernel_big = np.ones(
            (GRIPPER_GRASP_SAFE_WIDTH_PIXEL, GRIPPER_GRASP_INNER_DISTANCE_PIXEL), dtype=np.uint8
        )
        if (
            is_real
        ):  # due to color, depth sensor and lighting, the size of object looks a bit smaller.
            threshold_big = GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 5
            threshold_small = (
                GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 10
            )
        else:
            threshold_big = GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 10
            threshold_small = (
                GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 20
            )
        depth_heightmap_pad.shape = (depth_heightmap_pad.shape[0], depth_heightmap_pad.shape[1])
        for rotate_idx in range(len(grasp_predictions)):
            color_mask = utils.rotate(mask_pad, rotate_idx * (360.0 / NUM_ROTATION), True)
            color_mask[color_mask == 0] = 1
            color_mask[color_mask == 255] = 0
            no_target_mask = color_mask
            bg_mask = utils.rotate(mask_bg_pad, rotate_idx * (360.0 / NUM_ROTATION), True)
            no_target_mask[bg_mask == 255] = 0
            # only grasp one object
            invalid_mask = cv2.filter2D(no_target_mask, -1, kernel_big)
            invalid_mask = utils.rotate(invalid_mask, -rotate_idx * (360.0 / NUM_ROTATION), True)
            grasp_predictions[rotate_idx][invalid_mask > threshold_small] = (
                grasp_predictions[rotate_idx][invalid_mask > threshold_small] / 2
            )
            grasp_predictions[rotate_idx][invalid_mask > threshold_big] = 0

        # collision checking, only work for one level
        if post_checking:
            mask = cv2.inRange(temp, BG_THRESHOLD["low"], BG_THRESHOLD["high"])
            mask = 255 - mask
            mask_pad = np.pad(mask, IMAGE_PAD_WIDTH, "constant", constant_values=0)
            check_kernel = np.ones(
                (GRIPPER_GRASP_WIDTH_PIXEL, GRIPPER_GRASP_OUTER_DISTANCE_PIXEL), dtype=np.uint8
            )
            left_bound = math.floor(
                (GRIPPER_GRASP_OUTER_DISTANCE_PIXEL - GRIPPER_GRASP_INNER_DISTANCE_PIXEL) / 2
            )
            right_bound = (
                math.ceil(
                    (GRIPPER_GRASP_OUTER_DISTANCE_PIXEL + GRIPPER_GRASP_INNER_DISTANCE_PIXEL) / 2
                )
                + 1
            )
            check_kernel[:, left_bound:right_bound] = 0
            for rotate_idx in range(len(grasp_predictions)):
                object_mask = utils.rotate(mask_pad, rotate_idx * (360.0 / NUM_ROTATION), True)
                invalid_mask = cv2.filter2D(object_mask, -1, check_kernel)
                invalid_mask[invalid_mask > 5] = 255
                invalid_mask = utils.rotate(
                    invalid_mask, -rotate_idx * (360.0 / NUM_ROTATION), True
                )
                grasp_predictions[rotate_idx][invalid_mask > 128] = 0
        grasp_predictions = grasp_predictions[
            :, padding_width_start:padding_width_end, padding_width_start:padding_width_end
        ]

        best_pix_ind = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
        grasp_q_value = grasp_predictions[best_pix_ind]

        return grasp_q_value, best_pix_ind, grasp_predictions

    def get_prediction_vis(self, predictions, color_heightmap, best_pix_ind, is_push=False):

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
                        prediction_vis,
                        (int(best_pix_ind[2]), int(best_pix_ind[1])),
                        7,
                        (0, 0, 255),
                        2,
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
                                max(0, int(center[0]) - GRIPPER_GRASP_INNER_DISTANCE_PIXEL // 2),
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
                background_image = utils.rotate(
                    color_heightmap, rotate_idx * (360.0 / num_rotations)
                )
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

@torch.no_grad()
def _sampled_prediction_precise(env, model, actions, mask_image):
    model.pre_train = True
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dataset = LifelongEvalDataset(env, actions, mask_image)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=len(actions), shuffle=False, num_workers=0, drop_last=False
    )
    rot_angle, input_data = next(iter(data_loader))
    input_data = input_data.to(device)
    # get output
    output = model(input_data)
    output = output.cpu().numpy()
    rot_angle = rot_angle.numpy()

    out_q = []
    for idx, out in enumerate(output):
        out = utils.rotate(out[0], -rot_angle[idx])
        action = actions[idx]
        q = np.max(
            out[
                action[0][0] + IMAGE_PAD_WIDTH - 3 : action[0][0] + IMAGE_PAD_WIDTH + 4,
                action[0][1] + IMAGE_PAD_WIDTH - 3 : action[0][1] + IMAGE_PAD_WIDTH + 4,
            ]
        )
        out_q.append(q)

    return out_q

@torch.no_grad()
def from_maskrcnn(model, color_image, device, plot=False):
    """
    Use Mask R-CNN to do instance segmentation and output masks in binary format.
    Assume it works in real world
    """
    image = color_image.copy()
    image = TF.to_tensor(image)
    prediction = model([image.to(device)])[0]
    final_mask = np.zeros((720, 1280), dtype=np.uint8)
    labels = {}
    if plot:
        pred_mask = np.zeros((720, 1280), dtype=np.uint8)
    for idx, mask in enumerate(prediction["masks"]):
        # TODO, 0.9 can be tuned
        threshold = 0.7
        if prediction["scores"][idx] > threshold:
            # get mask
            img = mask[0].mul(255).byte().cpu().numpy()
            # img = cv2.GaussianBlur(img, (3, 3), 0)
            img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            # too small
            if np.sum(img == 255) < 100:
                continue
            # overlap IoU 70%
            if np.sum(np.logical_and(final_mask > 0, img == 255)) > np.sum(img == 255) * 3 / 4:
                continue
            fill_pixels = np.logical_and(final_mask == 0, img == 255)
            final_mask[fill_pixels] = idx + 1
            labels[(idx + 1)] = prediction["labels"][idx].cpu().item()
            if plot:
                pred_mask[img > 0] = prediction["labels"][idx].cpu().item() * 10
                cv2.imwrite(str(idx) + "mask.png", img)
    if plot:
        cv2.imwrite("pred.png", pred_mask)
    print("Mask R-CNN: %d objects detected" % (len(np.unique(final_mask)) - 1), prediction["scores"].cpu())
    return final_mask, labels
