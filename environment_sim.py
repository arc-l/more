import time
import glob
import os
import pybullet as pb
import pybullet_data
from pybullet_utils import bullet_client
import numpy as np
import cameras
from constants import PIXEL_SIZE, WORKSPACE_LIMITS


class Environment:
    def __init__(self, gui=True, time_step=1 / 240):
        """Creates environment with PyBullet.

        Args:
        gui: show environment with PyBullet's built-in display viewer
        time_step: PyBullet physics simulation step speed. Default is 1 / 240.
        """

        self.time_step = time_step
        self.gui = gui
        self.pixel_size = PIXEL_SIZE
        self.obj_ids = {"fixed": [], "rigid": []}
        self.agent_cams = cameras.RealSenseD455.CONFIG
        self.oracle_cams = cameras.Oracle.CONFIG
        self.bounds = WORKSPACE_LIMITS
        self.home_joints = np.array([0, -0.8, 0.5, -0.2, -0.5, 0]) * np.pi
        self.ik_rest_joints = np.array([0, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        self.drop_joints0 = np.array([0.5, -0.8, 0.5, -0.2, -0.5, 0]) * np.pi
        self.drop_joints1 = np.array([1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi

        # Start PyBullet.
        self._pb = bullet_client.BulletClient(connection_mode=pb.GUI if gui else pb.DIRECT)
        self._client_id = self._pb._client
        self._pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._pb.setTimeStep(time_step)

        if gui:
            target = self._pb.getDebugVisualizerCamera()[11]
            self._pb.resetDebugVisualizerCamera(
                cameraDistance=1.5, cameraYaw=90, cameraPitch=-25, cameraTargetPosition=target,
            )

    @property
    def is_static(self):
        """Return true if objects are no longer moving."""
        v = [
            np.linalg.norm(self._pb.getBaseVelocity(i, physicsClientId=self._client_id)[0])
            for i in self.obj_ids["rigid"]
        ]
        return all(np.array(v) < 5e-3)

    @property
    def info(self):
        """Environment info variable with object poses, dimensions, and colors."""

        info = {}  # object id : (position, rotation, dimensions)
        for obj_ids in self.obj_ids.values():
            for obj_id in obj_ids:
                pos, rot = self._pb.getBasePositionAndOrientation(
                    obj_id, physicsClientId=self._client_id
                )
                dim = self._pb.getVisualShapeData(obj_id, physicsClientId=self._client_id)[0][3]
                info[obj_id] = (pos, rot, dim)
        return info

    def add_object_id(self, obj_id, category="rigid"):
        """List of (fixed, rigid) objects in env."""
        self.obj_ids[category].append(obj_id)

    def remove_object_id(self, obj_id, category="rigid"):
        """List of (fixed, rigid) objects in env."""
        self.obj_ids[category].remove(obj_id)

    def save_objects(self):
        """Save states of all rigid objects. If this is unstable, could use saveBullet."""
        success = False
        while not success:
            success = self.wait_static()
        object_states = []
        for obj in self.obj_ids["rigid"]:
            pos, orn = self._pb.getBasePositionAndOrientation(obj)
            linVel, angVel = self._pb.getBaseVelocity(obj)
            object_states.append((pos, orn, linVel, angVel))
        return object_states

    def restore_objects(self, object_states):
        """Restore states of all rigid objects. If this is unstable, could use restoreState along with saveBullet."""
        for idx, obj in enumerate(self.obj_ids["rigid"]):
            pos, orn, linVel, angVel = object_states[idx]
            self._pb.resetBasePositionAndOrientation(obj, pos, orn)
            self._pb.resetBaseVelocity(obj, linVel, angVel)
        success = self.wait_static()
        return success

    def wait_static(self, timeout=3):
        """Step simulator asynchronously until objects settle."""
        self._pb.stepSimulation()
        t0 = time.time()
        while (time.time() - t0) < timeout:
            if self.is_static:
                return True
            self._pb.stepSimulation()
        print(f"Warning: Wait static exceeded {timeout} second timeout. Skipping.")
        return False

    def reset(self):
        self.obj_ids = {"fixed": [], "rigid": []}
        self.target_obj_id = -1
        self._pb.resetSimulation()
        self._pb.setGravity(0, 0, -9.8)

        # Temporarily disable rendering to load scene faster.
        if self.gui:
            self._pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)

        # Load workspace
        self.plane = self._pb.loadURDF(
            "plane.urdf", basePosition=(0, 0, -0.0005), useFixedBase=True,
        )
        self.workspace = self._pb.loadURDF(
            "assets/workspace/workspace.urdf", basePosition=(0.5, 0, 0), useFixedBase=True,
        )
        self._pb.changeDynamics(
            self.plane,
            -1,
            lateralFriction=1.1,
            restitution=0.5,
            linearDamping=0.5,
            angularDamping=0.5,
        )
        self._pb.changeDynamics(
            self.workspace,
            -1,
            lateralFriction=1.1,
            restitution=0.5,
            linearDamping=0.5,
            angularDamping=0.5,
        )

        # Load UR5e
        self.ur5e = self._pb.loadURDF(
            "assets/ur5e/ur5e.urdf", basePosition=(0, 0, 0), useFixedBase=True,
        )
        self.ur5e_joints = []
        for i in range(self._pb.getNumJoints(self.ur5e)):
            info = self._pb.getJointInfo(self.ur5e, i)
            joint_id = info[0]
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            if joint_name == "ee_fixed_joint":
                self.ur5e_ee_id = joint_id
            if joint_type == pb.JOINT_REVOLUTE:
                self.ur5e_joints.append(joint_id)
        self._pb.enableJointForceTorqueSensor(self.ur5e, self.ur5e_ee_id, 1)

        self.setup_gripper()

        # Move robot to home joint configuration.
        success = self.go_home()
        self.close_gripper()
        self.open_gripper()

        if not success:
            print("Simulation is wrong!")
            exit()

        # Re-enable rendering.
        if self.gui:
            self._pb.configureDebugVisualizer(
                self._pb.COV_ENABLE_RENDERING, 1, physicsClientId=self._client_id
            )

    def setup_gripper(self):
        """Load end-effector: gripper"""
        ee_position, _ = self.get_link_pose(self.ur5e, self.ur5e_ee_id)
        self.ee = self._pb.loadURDF(
            "assets/ur5e/gripper/robotiq_2f_85.urdf",
            ee_position,
            self._pb.getQuaternionFromEuler((0, -np.pi / 2, 0)),
        )
        self.ee_tip_offset = 0.1625
        self.gripper_angle_open = 0.03
        self.gripper_angle_close = 0.8
        self.gripper_angle_close_threshold = 0.73
        self.gripper_mimic_joints = {
            "left_inner_finger_joint": -1,
            "left_inner_knuckle_joint": -1,
            "right_outer_knuckle_joint": -1,
            "right_inner_finger_joint": -1,
            "right_inner_knuckle_joint": -1,
        }
        for i in range(self._pb.getNumJoints(self.ee)):
            info = self._pb.getJointInfo(self.ee, i)
            joint_id = info[0]
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            if joint_name == "finger_joint":
                self.gripper_main_joint = joint_id
            elif joint_name == "dummy_center_fixed_joint":
                self.ee_tip_id = joint_id
            elif "finger_pad_joint" in joint_name:
                self._pb.changeDynamics(
                    self.ee, joint_id, lateralFriction=0.9
                )
                self.ee_finger_pad_id = joint_id
            elif joint_type == pb.JOINT_REVOLUTE:
                self.gripper_mimic_joints[joint_name] = joint_id
                # Keep the joints static
                self._pb.setJointMotorControl2(
                    self.ee, joint_id, pb.VELOCITY_CONTROL, targetVelocity=0, force=0,
                )
        self.ee_constraint = self._pb.createConstraint(
            parentBodyUniqueId=self.ur5e,
            parentLinkIndex=self.ur5e_ee_id,
            childBodyUniqueId=self.ee,
            childLinkIndex=-1,
            jointType=pb.JOINT_FIXED,
            jointAxis=(0, 0, 1),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, -0.02),
            childFrameOrientation=pb.getQuaternionFromEuler((0, -np.pi / 2, 0)),
            physicsClientId=self._client_id,
        )
        self._pb.changeConstraint(self.ee_constraint, maxForce=10000)
        self._pb.enableJointForceTorqueSensor(self.ee, self.gripper_main_joint, 1)

        # Set up mimic joints in robotiq gripper: left
        c = self._pb.createConstraint(
            self.ee,
            self.gripper_main_joint,
            self.ee,
            self.gripper_mimic_joints["left_inner_finger_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        self._pb.changeConstraint(c, gearRatio=1, erp=0.8, maxForce=10000)
        c = self._pb.createConstraint(
            self.ee,
            self.gripper_main_joint,
            self.ee,
            self.gripper_mimic_joints["left_inner_knuckle_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        self._pb.changeConstraint(c, gearRatio=-1, erp=0.8, maxForce=10000)
        # Set up mimic joints in robotiq gripper: right
        c = self._pb.createConstraint(
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            self.ee,
            self.gripper_mimic_joints["right_inner_finger_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        self._pb.changeConstraint(c, gearRatio=1, erp=0.8, maxForce=10000)
        c = self._pb.createConstraint(
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            self.ee,
            self.gripper_mimic_joints["right_inner_knuckle_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        self._pb.changeConstraint(c, gearRatio=-1, erp=0.8, maxForce=10000)
        # Set up mimic joints in robotiq gripper: connect left and right
        c = self._pb.createConstraint(
            self.ee,
            self.gripper_main_joint,
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[0, 1, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
            physicsClientId=self._client_id,
        )
        self._pb.changeConstraint(c, gearRatio=-1, erp=0.8, maxForce=1000)

    def step(self, pose0=None, pose1=None):
        """Execute action with specified primitive.

        Args:
            action: action to execute.

        Returns:
            obs, done
        """
        if pose0 is not None and pose1 is not None:
            success = self.push(pose0, pose1)
            # Exit early if action times out.
            if not success:
                return {}, False

        # Step simulator asynchronously until objects settle.
        while not self.is_static:
            self._pb.stepSimulation()

        # Get RGB-D camera image observations.
        obs = {"color": [], "depth": []}
        for config in self.agent_cams:
            color, depth, _ = self.render_camera(config)
            obs["color"].append(color)
            obs["depth"].append(depth)

        return obs, True

    def seed(self, seed=None):
        self._random = np.random.RandomState(seed)
        return seed

    def render_camera(self, config):
        """Render RGB-D image with specified camera configuration."""

        # OpenGL camera settings.
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = pb.getMatrixFromQuaternion(config["rotation"])
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = config["position"] + lookdir
        focal_len = config["intrinsics"][0, 0]
        znear, zfar = config["zrange"]
        viewm = pb.computeViewMatrix(config["position"], lookat, updir)
        fovh = (config["image_size"][0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = config["image_size"][1] / config["image_size"][0]
        projm = pb.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

        # Render with OpenGL camera settings.
        _, _, color, depth, segm = self._pb.getCameraImage(
            width=config["image_size"][1],
            height=config["image_size"][0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            shadow=0,
            flags=pb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=pb.ER_BULLET_HARDWARE_OPENGL,
        )

        # Get color image.
        color_image_size = (config["image_size"][0], config["image_size"][1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        if config["noise"]:
            color = np.int32(color)
            color += np.int32(self._random.normal(0, 3, color.shape))
            color = np.uint8(np.clip(color, 0, 255))

        # Get depth image.
        depth_image_size = (config["image_size"][0], config["image_size"][1])
        zbuffer = np.array(depth).reshape(depth_image_size)
        depth = zfar + znear - (2.0 * zbuffer - 1.0) * (zfar - znear)
        depth = (2.0 * znear * zfar) / depth
        if config["noise"]:
            depth += self._random.normal(0, 0.003, depth_image_size)

        # Get segmentation image.
        segm = np.uint8(segm).reshape(depth_image_size)

        return color, depth, segm

    def __del__(self):
        self._pb.disconnect()

    def get_link_pose(self, body, link):
        result = self._pb.getLinkState(body, link)
        return result[4], result[5]

    def add_objects(self, num_obj, workspace_limits):
        """Randomly dropped objects to the workspace"""
        color_space = (
            np.asarray(
                [
                    [78.0, 121.0, 167.0],  # blue
                    [89.0, 161.0, 79.0],  # green
                    [156, 117, 95],  # brown
                    [242, 142, 43],  # orange
                    [237.0, 201.0, 72.0],  # yellow
                    [186, 176, 172],  # gray
                    [255.0, 87.0, 89.0],  # red
                    [176, 122, 161],  # purple
                    [118, 183, 178],  # cyan
                    [255, 157, 167],  # pink
                ]
            )
            / 255.0
        )
        mesh_list = glob.glob("assets/blocks/*.urdf")
        obj_mesh_ind = np.random.randint(0, len(mesh_list), size=num_obj)
        obj_mesh_color = color_space[np.asarray(range(num_obj)) % 10, :]

        # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
        body_ids = []
        with open("hard-cases/" + "temp.txt", "w") as out_file:
            for object_idx in range(len(obj_mesh_ind)):
                curr_mesh_file = mesh_list[obj_mesh_ind[object_idx]]
                drop_x = (
                    (workspace_limits[0][1] - workspace_limits[0][0] - 0.2) * np.random.random_sample()
                    + workspace_limits[0][0]
                    + 0.1
                )
                drop_y = (
                    (workspace_limits[1][1] - workspace_limits[1][0] - 0.2) * np.random.random_sample()
                    + workspace_limits[1][0]
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
                body_id = self._pb.loadURDF(
                    curr_mesh_file, object_position, self._pb.getQuaternionFromEuler(object_orientation),
                    flags=pb.URDF_ENABLE_SLEEPING
                )
                self._pb.changeVisualShape(body_id, -1, rgbaColor=object_color)
                body_ids.append(body_id)
                env.add_object_id(body_id)
                env.wait_static()

                out_file.write(
                    "%s %.18e %.18e %.18e %.18e %.18e %.18e %.18e %.18e %.18e\n"
                    % (
                        curr_mesh_file,
                        object_color[0],
                        object_color[1],
                        object_color[2],
                        object_position[0],
                        object_position[1],
                        object_position[2],
                        object_orientation[0],
                        object_orientation[1],
                        object_orientation[2],
                    )
                )

        return body_ids, True

    def add_object_push_from_file(self, file_name, switch=None):
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
        # Switch color of the first and the second for augmentation
        if switch is not None:
            temp = obj_mesh_colors[0]
            obj_mesh_colors[0] = obj_mesh_colors[switch]
            obj_mesh_colors[switch] = temp

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
            body_id = self._pb.loadURDF(
                curr_mesh_file,
                object_position,
                self._pb.getQuaternionFromEuler(object_orientation),
                flags=pb.URDF_ENABLE_SLEEPING
            )
            self._pb.changeVisualShape(body_id, -1, rgbaColor=object_color)
            self.add_object_id(body_id)
            if switch is not None:
                if switch == object_idx:
                    self.target_obj_id = body_id
            else:
                if object_idx == 0:
                    self.target_obj_id = body_id
            success &= self.wait_static()
            success &= self.wait_static()

        # give time to stop
        for _ in range(5):
            self._pb.stepSimulation()

        return success

    # ---------------------------------------------------------------------------
    # Robot Movement Functions
    # ---------------------------------------------------------------------------

    def go_home(self):
        return self.move_joints(self.home_joints)

    def move_joints(self, target_joints, speed=0.01, timeout=3):
        """Move UR5e to target joint configuration."""
        t0 = time.time()
        while (time.time() - t0) < timeout:
            current_joints = np.array(
                [
                    self._pb.getJointState(self.ur5e, i, physicsClientId=self._client_id)[0]
                    for i in self.ur5e_joints
                ]
            )
            pos, _ = self.get_link_pose(self.ee, self.ee_tip_id)
            if pos[2] < 0.005:
                print(f"Warning: move_joints tip height is {pos[2]}. Skipping.")
                return False
            diff_joints = target_joints - current_joints
            if all(np.abs(diff_joints) < 0.05):
                # give time to stop
                for _ in range(5):
                    self._pb.stepSimulation()
                return True

            # Move with constant velocity
            norm = np.linalg.norm(diff_joints)
            v = diff_joints / norm if norm > 0 else 0
            step_joints = current_joints + v * speed
            self._pb.setJointMotorControlArray(
                bodyIndex=self.ur5e,
                jointIndices=self.ur5e_joints,
                controlMode=pb.POSITION_CONTROL,
                targetPositions=step_joints,
                positionGains=np.ones(len(self.ur5e_joints)),
            )
            self._pb.stepSimulation()
        print(f"Warning: move_joints exceeded {timeout} second timeout. Skipping.")
        return False

    def move_ee_pose(self, pose, speed=0.01):
        """Move UR5e to target end effector pose."""
        target_joints = self.solve_ik(pose)
        return self.move_joints(target_joints, speed)

    def solve_ik(self, pose):
        """Calculate joint configuration with inverse kinematics."""
        joints = self._pb.calculateInverseKinematics(
            bodyUniqueId=self.ur5e,
            endEffectorLinkIndex=self.ur5e_ee_id,
            targetPosition=pose[0],
            targetOrientation=pose[1],
            lowerLimits=[-6.283, -6.283, -3.141, -6.283, -6.283, -6.283],
            upperLimits=[6.283, 6.283, 3.141, 6.283, 6.283, 6.283],
            jointRanges=[12.566, 12.566, 6.282, 12.566, 12.566, 12.566],
            restPoses=np.float32(self.ik_rest_joints).tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5,
        )
        joints = np.array(joints, dtype=np.float32)
        # joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        return joints

    def straight_move(self, pose0, pose1, rot, speed=0.01, max_force=300, detect_force=False, is_push=False):
        """Move every 1 cm, keep the move in a straight line instead of a curve. Keep level with rot"""
        step_distance = 0.01  # every 1 cm
        vec = np.float32(pose1) - np.float32(pose0)
        length = np.linalg.norm(vec)
        vec = vec / length
        n_push = np.int32(np.floor(length / step_distance))  # every 1 cm
        success = True
        for n in range(n_push):
            target = pose0 + vec * n * step_distance
            success &= self.move_ee_pose((target, rot), speed)
            if detect_force:
                force = np.sum(
                    np.abs(np.array(self._pb.getJointState(self.ur5e, self.ur5e_ee_id)[2]))
                )
                if force > max_force:
                    target = target - vec * 2 * step_distance
                    self.move_ee_pose((target, rot), speed)
                    print(f"Force is {force}, exceed the max force {max_force}")
                    return False    
        if is_push:
            speed /= 5
        success &= self.move_ee_pose((pose1, rot), speed)
        return success

    def push(self, pose0, pose1, speed=0.002, verbose=True):
        """Execute pushing primitive.

        Args:
            pose0: SE(3) starting pose.
            pose1: SE(3) ending pose.
            speed: the speed of the planar push.

        Returns:
            success: robot movement success if True.
        """

        # close the gripper
        self.close_gripper(is_slow=False)

        # Adjust push start and end positions.
        pos0 = np.array(pose0, dtype=np.float32)
        pos1 = np.array(pose1, dtype=np.float32)
        pos0[2] += self.ee_tip_offset
        pos1[2] += self.ee_tip_offset
        vec = pos1 - pos0
        length = np.linalg.norm(vec)
        vec = vec / length
        over0 = np.array((pos0[0], pos0[1], pos0[2] + 0.05))
        over0h = np.array((pos0[0], pos0[1], pos0[2] + 0.2))
        over1 = np.array((pos1[0], pos1[1], pos1[2] + 0.05))
        over1h = np.array((pos1[0], pos1[1], pos1[2] + 0.2))

        # Align against push direction.
        theta = np.arctan2(vec[1], vec[0]) + np.pi / 2
        rot = pb.getQuaternionFromEuler([np.pi / 2, np.pi / 2, theta])

        # Execute push.
        success = self.move_joints(self.ik_rest_joints)
        if success:
            success = self.move_ee_pose((over0h, rot))
        if success:
            success = self.straight_move(over0h, over0, rot, detect_force=True)
        if success:
            success = self.straight_move(over0, pos0, rot, detect_force=True)
        if success:
            success = self.straight_move(pos0, pos1, rot, speed, detect_force=True, is_push=True)
        if success:
            success = self.straight_move(pos1, over1, rot, speed)
        if success:
            success = self.straight_move(over1, over1h, rot)
        self.go_home()

        if verbose:
            print(f"Push from {pose0} to {pose1}, {success}")

        return success

    def grasp(self, pose, angle, speed=0.005):
        """Execute grasping primitive.

        Args:
            pose: SE(3) grasping pose.
            angle: rotation angle

        Returns:
            success: robot movement success if True.
        """

        # Handle unexpected behavior
        self._pb.changeDynamics(
            self.ee, self.ee_finger_pad_id, lateralFriction=0.9, spinningFriction=0.1
        )

        # Adjust grasp positions.
        pos = np.array(pose, dtype=np.float32)
        pos[2] = max(pos[2] - 0.04, self.bounds[2][0])
        pos[2] += self.ee_tip_offset

        # Align against grasp direction.
        angle = ((angle) % np.pi) - np.pi / 2
        rot = pb.getQuaternionFromEuler([np.pi / 2, np.pi / 2, -angle])

        over = np.array((pos[0], pos[1], pos[2] + 0.2))

        # Execute push.
        self.open_gripper()
        success = self.move_joints(self.ik_rest_joints)
        if success:
            success = self.move_ee_pose((over, rot))
        if success:
            success = self.straight_move(over, pos, rot, speed, detect_force=True)
        if success:
            self.close_gripper()
            success = self.straight_move(pos, over, rot, speed)
            success &= self.is_gripper_closed
        if success:
            success = self.move_joints(self.drop_joints1)
            success &= self.is_gripper_closed
            self.open_gripper(is_slow=True)
        self.go_home()

        print(f"Grasp at {pose}, the grasp {success}")

        self._pb.changeDynamics(
            self.ee, self.ee_finger_pad_id, lateralFriction=0.9
        )

        return success

    def open_gripper(self, is_slow=False):
        self._move_gripper(self.gripper_angle_open, is_slow=is_slow)

    def close_gripper(self, is_slow=True):
        self._move_gripper(self.gripper_angle_close, is_slow=is_slow)

    @property
    def is_gripper_closed(self):
        gripper_angle = self._pb.getJointState(
            self.ee, self.gripper_main_joint, physicsClientId=self._client_id
        )[0]
        return gripper_angle < self.gripper_angle_close_threshold

    def _move_gripper(self, target_angle, timeout=3, is_slow=False):
        t0 = time.time()
        prev_angle = self._pb.getJointState(
            self.ee, self.gripper_main_joint, physicsClientId=self._client_id
        )[0]

        if is_slow:
            self._pb.setJointMotorControl2(
                self.ee,
                self.gripper_main_joint,
                pb.VELOCITY_CONTROL,
                targetVelocity=1 if target_angle > 0.5 else -1,
                maxVelocity=1 if target_angle > 0.5 else -1,
                force=3,
                physicsClientId=self._client_id,
            )
            self._pb.setJointMotorControl2(
                self.ee,
                self.gripper_mimic_joints["right_outer_knuckle_joint"],
                pb.VELOCITY_CONTROL,
                targetVelocity=1 if target_angle > 0.5 else -1,
                maxVelocity=1 if target_angle > 0.5 else -1,
                force=3,
                physicsClientId=self._client_id,
            )
            for _ in range(10):
                self._pb.stepSimulation()
            while (time.time() - t0) < timeout:
                current_angle = self._pb.getJointState(self.ee, self.gripper_main_joint)[0]
                diff_angle = abs(current_angle - prev_angle)
                if diff_angle < 1e-4:
                    break
                prev_angle = current_angle
                for _ in range(10):
                    self._pb.stepSimulation()
        # maintain the angles
        self._pb.setJointMotorControl2(
            self.ee,
            self.gripper_main_joint,
            pb.POSITION_CONTROL,
            targetPosition=target_angle,
            force=3.1,
        )
        self._pb.setJointMotorControl2(
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            pb.POSITION_CONTROL,
            targetPosition=target_angle,
            force=3.1,
        )
        for _ in range(10):
            self._pb.stepSimulation()


if __name__ == "__main__":
    env = Environment()
    env.reset()

    print(pb.getPhysicsEngineParameters(env._client_id))

    time.sleep(1)
    # env.add_object_push_from_file("hard-cases/temp.txt", switch=None)

    # push_start = [4.280000000000000471e-01, -3.400000000000000244e-02, 0.01]
    # push_end = [5.020000000000000018e-01, -3.400000000000000244e-02, 0.01]
    # env.push(push_start, push_end)
    # time.sleep(1)

    env.render_camera(env.oracle_cams[0])

    for i in range(16):
        best_rotation_angle = np.deg2rad(90 - i * (360.0 / 16))
        primitive_position = [0.6, 0, 0.01]
        primitive_position_end = [
            primitive_position[0] + 0.1 * np.cos(best_rotation_angle),
            primitive_position[1] + 0.1 * np.sin(best_rotation_angle),
            0.01,
        ]
        env.push(primitive_position, primitive_position_end, speed=0.0002)
        env._pb.addUserDebugLine(primitive_position, primitive_position_end, lifeTime=0)

        # angle = np.deg2rad(i * 360 / 16)
        # pos = [0.5, 0, 0.05]
        # env.grasp(pos, angle)

        time.sleep(1)
