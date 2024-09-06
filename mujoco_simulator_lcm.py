import lcm
from lcm_package.SimCommand import SimCommand
from lcm_package.SimState import SimState
from lcm_package.state_estimator_lcmt import state_estimator_lcmt
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import time
import math
import threading
from scipy.spatial.transform import Rotation as R
# from ros_pub import RosPublisher
# import rospy


class MujocoSimulator:
    def __init__(self, model_file) -> None:
        self.display_state_estimator = False
        self.fix_base = 1
        self.xml_path = model_file
        self.default_joint_pos = [0]*12
        self.print_camera_config = 1
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.last_mouse_posx = 0
        self.last_mouse_posy = 0
        self.first_run_mujoco_step = True
        # parameters for simulation control
        self.sim_time = 0
        # ******************settings for lcm
        self.lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=254")
        self.command = SimCommand()
        self.state = SimState()
        sub = self.lc.subscribe("sim_command", self.lcmCmdCallback)
        sub.set_queue_capacity(10)  # store only one data
        
        # if self.display_state_estimator:
        self.state_est = state_estimator_lcmt()
        sub = self.lc.subscribe("state_estimator", self.lcmStateEstCallback)
        sub.set_queue_capacity(10)  # store only one data
        
        self.lcm_thread = threading.Thread(target=self.lcmHandleThread)
        self.lcm_pub_thread = threading.Thread(target=self.lcmPublish)

    def lcmCmdCallback(self, channel, data):
        self.command = SimCommand.decode(data)
        self.simulationStep()

    def lcmStateEstCallback(self, channel, data):
        self.state_est = state_estimator_lcmt.decode(data)

    def lcmPublish(self):
        while (True):
            begin = time.time()
            self.updateState()
            self.lc.publish("sim_state", self.state.encode())
            end = time.time()
            # print("lcm pub time:",end-begin)
            while (end-begin < 0.001):
                time.sleep(0.0001)
                end = time.time()

    def lcmHandleThread(self):
        while (True):
            self.lc.handle()

    def resetSim(self):
        # mj.mj_resetData(self.model, self.data)
        if self.fix_base:
            self.data.qpos = self.default_joint_pos.copy()
            self.data.qvel[0:12] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            self.data.qpos[0:7] = [-3, 0, 1.2, 1, 0, 0, 0]
            self.data.qpos[7:19] = self.default_joint_pos.copy()
            self.data.qvel[0:18] = [0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        mj.mj_forward(self.model, self.data)
        mj.mj_step(self.model, self.data)

    # set motor's control mode to be position mode
    def setPostionServo(self, actuator_no, kp):
        self.model.actuator_gainprm[actuator_no, 0:3] = [kp, 0, 0]
        self.model.actuator_biasprm[actuator_no, 0:3] = [0, -kp, 0]
        self.model.actuator_biastype[actuator_no] = 1

    # set motor's control mode to be velocity mode
    def setVelocityServo(self, actuator_no, kv):
        self.model.actuator_gainprm[actuator_no, 0:3] = [kv, 0, 0]
        self.model.actuator_biasprm[actuator_no, 0:3] = [0, 0, -kv]
        self.model.actuator_biastype[actuator_no] = 1

    def setTorqueServo(self, actuator_no):  # set motor's control mode to be torque mode
        self.model.actuator_gainprm[actuator_no, 0:3] = [1, 0, 0]
        self.model.actuator_biasprm[actuator_no, 0:3] = [0, 0, 0]
        self.model.actuator_biastype[actuator_no] = 0

    def initSimulator(self):
        # MuJoCo data structures
        self.model = mj.MjModel.from_xml_path(self.xml_path)  # MuJoCo model
        self.data = mj.MjData(self.model)                # MuJoCo data
        self.cam = mj.MjvCamera()                        # Abstract camera
        self.opt = mj.MjvOption()                        # visualization options
        self.opt.flags[14] = 1  # show contact area
        self.opt.flags[15] = 1  # show contact force
        # set camera configuration
        self.cam.azimuth = 90
        self.cam.elevation = -45
        self.cam.distance = 2
        self.cam.lookat = np.array([0.0, 0.0, 0])
        # print(self.data.qpos)
        if (self.data.qpos.size == 38 or self.data.qpos.size == 19):  # float base model
            self.data.qpos[7:19] = self.default_joint_pos.copy()
            self.fix_base = 0
        elif (self.data.qpos.size == 31 or self.data.qpos.size == 24 or self.data.qpos.size == 12):  # fixed base model
            self.fix_base = 1
            self.data.qpos[0:12] = self.default_joint_pos.copy()
        else:
            print('please check dof')
        
        if (self.data.qpos.size > 19):
            self.display_state_estimator = True
        # Init GLFW library, create window, make OpenGL context current, request v-sync
        glfw.init()
        self.window = glfw.create_window(1000, 900, "Demo", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        # initialize visualization data structures
        mj.mjv_defaultCamera(self.cam)
        mj.mjv_defaultOption(self.opt)
        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.context = mj.MjrContext( self.model, mj.mjtFontScale.mjFONTSCALE_150.value)
        # print camera configuration (help to initialize the view)
        if (self.print_camera_config == 1):
            print('cam.azimuth =', self.cam.azimuth, ';', 'cam.elevation =',
                  self.cam.elevation, ';', 'cam.distance = ', self.cam.distance)
            print('cam.lookat =np.array([', self.cam.lookat[0], ',',
                  self.cam.lookat[1], ',', self.cam.lookat[2], '])')
        for i in range(12):
            self.setTorqueServo(i)
        total_mass = sum(self.model.body_mass[1:14])
        print("total mass:", total_mass)
        # self.setVelocityServo(4, 10)
        # self.setTorqueServo(7)
        glfw.set_key_callback(self.window, self.keyboard)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move)
        glfw.set_mouse_button_callback(self.window, self.mouse_button)
        glfw.set_scroll_callback(self.window, self.mouse_scroll)
        mj.mj_forward(self.model, self.data)
        #

    def controller(self, model, data):
        cmd = self.command
        for i in range(12):
            if (self.fix_base):
                self.data.ctrl[i] = cmd.joint_stiffness[i]*(cmd.joint_position[i]-self.data.qpos[i])+cmd.joint_damping[i]*(
                    cmd.joint_velocity[i]-self.data.qvel[i])+cmd.joint_feed_forward_torque[i]
            else:
                self.data.ctrl[i] = cmd.joint_stiffness[i]*(cmd.joint_position[i]-self.data.qpos[7+i])+cmd.joint_damping[i]*(
                    cmd.joint_velocity[i]-self.data.qvel[6+i])+cmd.joint_feed_forward_torque[i]

    def simulationStep(self):
        if self.first_run_mujoco_step:
            mj.set_mjcb_control(self.controller)
            self.first_run_mujoco_step = False
        self.sim_time = time.time()
        mj.mj_step(self.model, self.data)
        self.updateState()
        
        # display state estimator
        if self.display_state_estimator:
            self.displayStateEstimator()
            
        # publish data through lcm
        # self.lc.publish("simulator_state", self.state.encode())
        
    def displayStateEstimator(self):
        if any(np.isnan(self.state_est.p)) or any(np.isnan(self.state_est.quat)):  # check isnan
            # print("state_est is nan",)
            a = 1
        else:
            if self.fix_base:
                self.data.qpos[19-7:22-7] = self.state_est.p[0:3]
                self.data.qpos[22-7:26-7] = self.state_est.quat[0:4]
                self.data.qpos[26-7:38-7] = self.data.qpos[7-7:19-7]
            else:
                self.data.qpos[19:22] = self.state_est.p[0:3]
                self.data.qpos[22:26] = self.state_est.quat[0:4]
                self.data.qpos[26:38] = self.data.qpos[7:19]
        mj.mj_forward(self.model, self.data)
        
        
    def runSimulation(self):
        self.lcm_thread.start()
        self.lcm_pub_thread.start()
        # key and mouse control
        glfw.set_key_callback(self.window, self.keyboard)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move)
        glfw.set_mouse_button_callback(self.window, self.mouse_button)
        glfw.set_scroll_callback(self.window, self.mouse_scroll)
        mj.mj_forward(self.model, self.data)

        # counter=0
        while not glfw.window_should_close(self.window):
            # mj.mj_forward(self.model, self.data)
            time_prev = self.data.time
            viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
            viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
        # Update scene and render
            if self.fix_base:
                self.cam.lookat = np.array([0, 0, 1.2])
            else:
                self.cam.lookat = np.array(self.data.qpos[0:3])
            self.opt.flags[14] = 1  # show contact area
            self.opt.flags[15] = 1  # show contact force
            mj.mjv_updateScene(self.model, self.data, self.opt, None,
                               self.cam, mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            mj.mjr_render(viewport, self.scene, self.context)
        # swap OpenGL buffers (blocking call due to v-sync)
            glfw.swap_buffers(self.window)
        # process pending GUI events, call GLFW callbacks
            glfw.poll_events()
            while (self.data.time-time_prev < 1.0/60.0):
                # mj.mj_step(self.model, self.data)
                time.sleep(0.0002)
        glfw.terminate()

    def updateState(self):
        # imu
        self.state.num_ranges=12
        self.state.imu_sensor_linear_acceleration = self.data.sensordata[7:10]  # include gravity 9.81
        self.state.imu_sensor_angular_velocity = self.data.sensordata[4:7]
        self.state.imu_sensor_quaternion = [self.data.sensordata[0], self.data.sensordata[1],
                               self.data.sensordata[2], self.data.sensordata[3]]
        # encoder
        if self.fix_base:
            self.state.joint_position =self.data.qpos[0:12]
            self.state.joint_velocity = self.data.qvel[0:12]
            self.state.joint_torque=self.data.qfrc_actuator[0:12]
            self.state.base_link_position = [0, 0, 0.8]
            self.state.base_link_linear_velocity = [0, 0, 0]
        else:
            self.state.joint_position = self.data.qpos[7:19]
            self.state.joint_velocity = self.data.qvel[6:18]
            self.state.joint_torque=self.data.qfrc_actuator[6:18]
            self.state.base_link_position = [self.data.qpos[0], self.data.qpos[1], self.data.qpos[2]]
            self.state.base_link_linear_velocity = [self.data.qvel[0], self.data.qvel[1], self.data.qvel[2]]
                # contact force
        # self.contact_force = np.array(self.data.sensordata[22:26])
        # # print("contact_force",self.contact_force)   
            

    def keyboard(self, window, key, scancode, act, mods):
        if (act == glfw.PRESS and key == glfw.KEY_R):
            self.resetSim()
        if act == glfw.PRESS and key == glfw.KEY_S:
            print('Pressed key s')
        if act == glfw.PRESS and key == glfw.KEY_UP:
            print('Pressed key up')
        if act == glfw.PRESS and key == glfw.KEY_DOWN:
            print('Pressed key down')
        if act == glfw.PRESS and key == glfw.KEY_LEFT:
            print('Pressed key left')
        if act == glfw.PRESS and key == glfw.KEY_RIGHT:
            print('Pressed key right')
    # update button state

    def mouse_button(self, window, button, act, mods):
        self.button_left = (glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
        self.button_middle = (glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
        self.button_right = (glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
        glfw.get_cursor_pos(window)  # update mouse position

    def mouse_scroll(self, window, xoffset, yoffset):
        action = mj.mjtMouse.mjMOUSE_ZOOM
        mj.mjv_moveCamera(self.model, action, 0.0, -0.05 *
                          yoffset, self.scene, self.cam)

    def mouse_move(self, window, xpos, ypos):
        # compute mouse displacement, save
        dx = xpos - self.last_mouse_posx
        dy = ypos - self.last_mouse_posy
        self.last_mouse_posx = xpos
        self.last_mouse_posy = ypos
        # # determine action based on mouse button
        if self.button_right:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
        elif self.button_left:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
        elif self.button_middle:
            action = mj.mjtMouse.mjMOUSE_ZOOM
        else:
            return
        width, height = glfw.get_window_size(window)  # get current window size
        mj.mjv_moveCamera(self.model, action, dx/height,
                          dy/height, self.scene, self.cam)


if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    abspath = os.path.join(dirname + "/model/zq_sa01/scene.xml")
    model_xml = abspath
    sim = MujocoSimulator(model_xml) 
    sim.initSimulator()
    sim.runSimulation()
