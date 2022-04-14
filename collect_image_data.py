import time
import datetime
import os
import glob
import pybullet as p
import numpy as np
import cv2
import utils

from environment_sim import Environment
from constants import (
    DEPTH_MIN,
    GRIPPER_PUSH_RADIUS_PIXEL,
    GRIPPER_PUSH_RADIUS_SAFE_PIXEL,
    IMAGE_SIZE,
    WORKSPACE_LIMITS,
    REAL_COLOR_SPACE,
    MODEL_MASK_ID,
)



class ImageDataCollector:
    def __init__(self, start_iter=0, end_iter=2000, base_directory=None, seed=0):
        self.rng = np.random.default_rng(seed)
        self.depth_min = DEPTH_MIN
        self.mesh_list = glob.glob("assets/blocks/*.urdf")
        self.mesh_list = [mesh for mesh in self.mesh_list if mesh.split("/")[-1] in MODEL_MASK_ID]

        # Create directory to save data
        timestamp = time.time()
        timestamp_value = datetime.datetime.fromtimestamp(timestamp)
        if base_directory is None:
            self.base_directory = os.path.join(
                os.path.abspath("logs_image"), timestamp_value.strftime("%Y-%m-%d-%H-%M-%S")
            )
        else:
            self.base_directory = base_directory
        print("Creating data logging session: %s" % (self.base_directory))
        self.color_heightmaps_directory = os.path.join(
            self.base_directory, "data", "color-heightmaps"
        )
        self.depth_heightmaps_directory = os.path.join(
            self.base_directory, "data", "depth-heightmaps"
        )
        self.mask_directory = os.path.join(self.base_directory, "data", "masks")

        if not os.path.exists(self.color_heightmaps_directory):
            os.makedirs(self.color_heightmaps_directory)
        if not os.path.exists(self.depth_heightmaps_directory):
            os.makedirs(self.depth_heightmaps_directory)
        if not os.path.exists(self.mask_directory):
            os.makedirs(self.mask_directory)

        self.iter = start_iter
        self.end_iter = end_iter

    def reset_np_random(self, seed):
        self.rng = np.random.default_rng(seed)

    def save_heightmaps(
        self,
        iteration,
        color_heightmap,
        depth_heightmap,
    ):
        color_heightmap = cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            os.path.join(self.color_heightmaps_directory, "%07d.color.png" % (iteration)),
            color_heightmap,
        )
        depth_heightmap = np.round(depth_heightmap * 100000).astype(
            np.uint16
        )  # Save depth in 1e-5 meters
        cv2.imwrite(
            os.path.join(self.depth_heightmaps_directory, "%07d.depth.png" % (iteration)),
            depth_heightmap,
        )

    def save_masks(self, iteration, mask):
        cv2.imwrite(os.path.join(self.mask_directory, "%07d.mask.png" % (iteration)), mask)

    def add_objects(self, env, num_obj):
        """Randomly dropped objects to the workspace"""
        obj_mesh_ind = self.rng.integers(0, len(self.mesh_list), size=num_obj)
        # sim color
        # obj_mesh_color = COLOR_SPACE[np.asarray(range(num_obj)) % len(COLOR_SPACE), :]
        # real color
        obj_mesh_color = self.rng.choice(REAL_COLOR_SPACE, size=4)
        new_obj_mesh_color = []
        for object_color in obj_mesh_color:
            new_obj_mesh_color.append([
                max(0, min(1, object_color[0] + self.rng.random() * 0.3 - 0.15)),
                max(0, min(1, object_color[1] + self.rng.random() * 0.3 - 0.15)),
                max(0, min(1, object_color[2] + self.rng.random() * 0.3 - 0.15)),
            ])
        obj_mesh_color = self.rng.choice(new_obj_mesh_color, size=num_obj)

        # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
        body_ids = []
        body_mask_ids = []
        for object_idx in range(len(obj_mesh_ind)):
            curr_mesh_file = self.mesh_list[obj_mesh_ind[object_idx]]
            drop_x = (
                (WORKSPACE_LIMITS[0][1] - WORKSPACE_LIMITS[0][0] - 0.2) * np.random.random_sample()
                + WORKSPACE_LIMITS[0][0]
                + 0.1
            )
            drop_y = (
                (WORKSPACE_LIMITS[1][1] - WORKSPACE_LIMITS[1][0] - 0.2) * np.random.random_sample()
                + WORKSPACE_LIMITS[1][0]
                + 0.1
            )
            object_position = [drop_x, drop_y, 0.2]
            object_orientation = [
                2 * np.pi * np.random.random_sample(),
                2 * np.pi * np.random.random_sample(),
                2 * np.pi * np.random.random_sample(),
            ]
            object_color = [
                obj_mesh_color[object_idx][0],
                obj_mesh_color[object_idx][1],
                obj_mesh_color[object_idx][2],
                1,
            ]
            body_id = p.loadURDF(
                curr_mesh_file, object_position, p.getQuaternionFromEuler(object_orientation)
            )
            body_mask_ids.append(MODEL_MASK_ID[curr_mesh_file.split("/")[-1]])
            p.changeVisualShape(body_id, -1, rgbaColor=object_color)
            body_ids.append(body_id)
            env.add_object_id(body_id)
            env.wait_static()

        return body_ids, body_mask_ids

    def add_object_push_from_file(self, env, file_name):
        body_ids = []
        success = True
        # Read data
        with open(file_name, "r") as preset_file:
            file_content = preset_file.readlines()
            num_obj = len(file_content)
            obj_files = []
            obj_mesh_colors = []
            obj_positions = []
            obj_orientations = []
            for object_idx in range(num_obj):
                file_content_curr_object = file_content[object_idx].split()
                obj_file = os.path.join("assets", "blocks", file_content_curr_object[0])
                obj_files.append(obj_file)
                obj_positions.append(
                    [
                        float(file_content_curr_object[4]),
                        float(file_content_curr_object[5]),
                        float(file_content_curr_object[6]),
                    ]
                )
                obj_orientations.append(
                    [
                        float(file_content_curr_object[7]),
                        float(file_content_curr_object[8]),
                        float(file_content_curr_object[9]),
                    ]
                )
                obj_mesh_colors.append(
                    [
                        float(file_content_curr_object[1]),
                        float(file_content_curr_object[2]),
                        float(file_content_curr_object[3]),
                    ]
                )
            

        # real color, ignore the color in file
        obj_mesh_color = self.rng.choice(REAL_COLOR_SPACE, size=4)
        new_obj_mesh_color = []
        for object_color in obj_mesh_color:
            new_obj_mesh_color.append([
                max(0.01, min(1, object_color[0] + self.rng.random() * 0.3 - 0.15)),
                max(0.01, min(1, object_color[1] + self.rng.random() * 0.3 - 0.15)),
                max(0.01, min(1, object_color[2] + self.rng.random() * 0.3 - 0.15)),
            ])
        obj_mesh_colors = self.rng.choice(new_obj_mesh_color, size=num_obj)
        body_mask_ids = []
        # Import objects
        for object_idx in range(num_obj):
            curr_mesh_file = obj_files[object_idx]
            object_position = [
                obj_positions[object_idx][0],
                obj_positions[object_idx][1],
                obj_positions[object_idx][2],
            ]
            object_orientation = [
                obj_orientations[object_idx][0],
                obj_orientations[object_idx][1],
                obj_orientations[object_idx][2],
            ]
            object_color = [
                obj_mesh_colors[object_idx][0],
                obj_mesh_colors[object_idx][1],
                obj_mesh_colors[object_idx][2],
                1,
            ]
            body_id = p.loadURDF(
                curr_mesh_file, object_position, p.getQuaternionFromEuler(object_orientation)
            )
            p.changeVisualShape(body_id, -1, rgbaColor=object_color)
            body_ids.append(body_id)
            body_mask_ids.append(MODEL_MASK_ID[curr_mesh_file.split("/")[-1]])
            env.add_object_id(body_id)
            success &= env.wait_static()
        success &= env.wait_static()

        return body_ids, body_mask_ids, success

    def get_push_action(self, depth):
        """Find target and push, the robot makes a push from left to right."""
        depth_heightmap = np.copy(depth)
        depth_heightmap[depth_heightmap <= self.depth_min] = 0
        depth_heightmap[depth_heightmap > self.depth_min] = 1

        y_indices = np.argwhere(depth_heightmap == 1)[:, 1]  # Find the y range
        if len(y_indices) == 0:
            print("find Skip")
            return None
        y_list_unique, y_list_count = np.unique(y_indices, return_counts=True)
        y_list_dist = y_list_count / y_list_count.sum()
        y = self.rng.choice(y_list_unique, p=y_list_dist)
        x_indices = np.argwhere(depth_heightmap[:, y] == 1)[:, 0]  # Find the x range
        x_indices_left = np.argwhere(
            depth_heightmap[:, max(0, y - GRIPPER_PUSH_RADIUS_PIXEL)] == 1
        )[
            :, 0
        ]  # Find the x range
        x_indices_right = np.argwhere(
            depth_heightmap[:, min(y + GRIPPER_PUSH_RADIUS_PIXEL, IMAGE_SIZE - 1)] == 1
        )[
            :, 0
        ]  # Find the x range
        if len(x_indices) == 0:
            print("Skip 1")
            return None
        x = x_indices.min()
        if len(x_indices_left) != 0:
            x = min(x, x_indices_left.min())
        if len(x_indices_right) != 0:
            x = min(x, x_indices_right.min())
        x = x - GRIPPER_PUSH_RADIUS_SAFE_PIXEL
        if x <= 0:
            print("Skip 2")
            return None

        safe_z_position = 0.01
        return [
            x * env.pixel_size + env.bounds[0][0],
            y * env.pixel_size + env.bounds[1][0],
            safe_z_position,
        ]


if __name__ == "__main__":

    env = Environment(gui=False)
    collector = ImageDataCollector(start_iter=0, end_iter=2000)
    cases = sorted(glob.glob("hard-cases/*.txt"))
    cases_idx = 0
    num_cases = len(cases)
    seed = 0

    # multi_thread_start = 1800
    # collector.iter += multi_thread_start
    # multi_thread_end = collector.iter + 300
    # seed += multi_thread_start

    while collector.iter < collector.end_iter:
        # if collector.iter > multi_thread_end:
        #     break

        print(f"-----Collecting: {collector.iter + 1}/{collector.end_iter}-----")
        collector.reset_np_random(seed)
        env.seed(seed)
        env.reset()
        # add objects
        # num_objs = collector.rng.integers(4, 12, size=1)[0]
        # body_ids, body_mask_ids = collector.add_objects(env, num_objs)
        body_ids, body_mask_ids, success = collector.add_object_push_from_file(env, cases[cases_idx])
        cases_idx += 1
        if cases_idx == num_cases:
            cases_idx = 0
        success = env.wait_static()
        if success:
            # record info0
            _, depth0p, _ = utils.get_true_heightmap(env)
            color0, depth0, segm0 = env.render_camera(env.agent_cams[0])
            # save data
            collector.save_heightmaps(collector.iter, color0, depth0)
            new_segm = np.zeros_like(segm0, dtype=np.uint16)
            for idx, body_id in enumerate(body_ids):
                new_segm[segm0 == body_id] = body_mask_ids[idx] + body_id
            print(np.unique(new_segm))
            collector.save_masks(collector.iter, new_segm)

            # push and save again
            # action = collector.get_push_action(depth0p)
            # if action is not None:
            #     action_end = [action[0] + PUSH_DISTANCE, action[1], action[2]]
            #     success = env.push(action, action_end)
            #     success &= env.wait_static()
            #     if success:
            #         # record info0
            #         # color0, depth0, segm0 = utils.get_true_heightmap(env)
            #         color0, depth0, segm0 = env.render_camera(env.agent_cams[0])
            #         # save data
            #         collector.save_heightmaps(collector.iter + collector.end_iter, color0, depth0)
            #         new_segm = np.zeros_like(segm0, dtype=np.uint16)
            #         for idx, body_id in enumerate(body_ids):
            #             new_segm[segm0 == body_id] = body_mask_ids[idx] + body_id
            #         print(np.unique(new_segm))
            #         collector.save_masks(collector.iter + collector.end_iter, new_segm)

            collector.iter += 1
        seed += 1
