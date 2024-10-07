
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import time
import math
import threading
from scipy.spatial.transform import Rotation as R
from inverse_kinematics import Kinematics as Kine
from trajectory import trapezoid_planning, PlotPositionAndVelocity
# from ros_pub import RosPublisher
# import rospy


class MujocoSimulator:
    def __init__(self, xml_file,urdf_file) -> None:
        self.display_state_estimator = False
        self.fix_base = 1
        self.xml_path = xml_file
        self.default_joint_pos = [0]*6
        self.print_camera_config = 1
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.last_mouse_posx = 0
        self.last_mouse_posy = 0
        self.first_run_mujoco_step = True
        # parameters for simulation control
        self.sim_time = 0
        self.qpos_cmd=[0]*6
        self.qvel_cmd=[0]*6
        self.kp=[500,1000,200,50,20,10]
        self.kd=[50,200,20,20,20,10]
        self.kine=Kine(urdf_file)
        # parameters for trajectory
        self.max_vel=1.0
        self.max_acc=2.0
        
        self.pos_array=np.array([[-280,  336,   673, 673, 1739, 1739, 1739,  2674, 2674, 3403, 3628, 3628, 3628],
                                 [-221, -221,  -147,-147,-147, -147, -147,  -147, -147, -147, -388, -344, -344],
                                 [834.7 ,834.7, 700, 453, 807,  453,  117.6, 807,  658,  410,  1352, 954,  477]]).dot(0.001)
        #set base position and orietation
        self.base_pos = np.array([0.1, 0, 0.4])
        self.base_ori=np.array([[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]])
        self.qpos_upper_limit = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).dot(3.1415926)
        self.qpos_lower_limit = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]).dot(3.1415926)
        self.finished=False

        


    def resetSim(self):
        # mj.mj_resetData(self.model, self.data)
        if self.fix_base:
            self.data.qpos = self.default_joint_pos.copy()
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
        self.cam.distance = 3
        self.cam.lookat = np.array([0.0, 0.0, 0.5])
        # print(self.data.qpos)
        self.data.qpos[0:6] = self.default_joint_pos.copy()
        
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
        # for i in range(6):
        #     self.setPostionServo(i, 200)
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
        if(self.first_run_mujoco_step):
            self.data.qpos=self.qpos_cmd
            mj.mj_forward(self.model, self.data)
            self.first_run_mujoco_step = False
        for i in range(6):
            self.data.ctrl[i]=self.kp[i]*(self.qpos_cmd[i]-self.data.qpos[i])+self.kd[i]*(self.qvel_cmd[i]-self.data.qvel[i])

        
    def runSimulation(self):
        # key and mouse control
        glfw.set_key_callback(self.window, self.keyboard)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move)
        glfw.set_mouse_button_callback(self.window, self.mouse_button)
        glfw.set_scroll_callback(self.window, self.mouse_scroll)
        mj.mj_forward(self.model, self.data)

        # counter=0
        start_time=self.data.time
        itr=0
        line_number=0
        while not glfw.window_should_close(self.window):
            # mj.mj_forward(self.model, self.data)
            time_prev = self.data.time
            viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
            viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
            # Update scene and render
            self.cam.lookat = np.array([0, 0, 0.8])
            self.cam.distance = 3
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
                if(line_number>=self.pos_array.shape[1]-8):
                    line_number=self.pos_array.shape[1]-8
                    self.finished=True
                start_pos=self.base_ori.T@(self.pos_array[:,line_number]-self.base_pos)
                end_pos=self.base_ori.T@(self.pos_array[:,line_number+1]-self.base_pos)
                if(line_number%2==0):
                    start_pos=self.base_ori.T@(self.pos_array[:,0]-self.base_pos)
                    end_pos=self.base_ori.T@(self.pos_array[:,1]-self.base_pos)
                else:
                    start_pos=self.base_ori.T@(self.pos_array[:,1]-self.base_pos)
                    end_pos=self.base_ori.T@(self.pos_array[:,0]-self.base_pos)
                t=itr*0.002
                cart_pos,cart_vel=trapezoid_planning(start_pos,end_pos,self.max_vel, self.max_acc, t)
                if(np.linalg.norm(cart_pos-end_pos)<1e-6 and not self.finished):
                    line_number+=1
                    itr=0

                # print("line_number",line_number)
                # print("cart_pos-end_pos",cart_pos-end_pos)

                qref=self.data.qpos
                omega_des=np.zeros(3)
                rotm=np.eye(3)
                self.qpos_cmd,self.qvel_cmd=self.kine.ikine(qref,cart_pos,rotm,cart_vel,omega_des,self.qpos_upper_limit,self.qpos_lower_limit)
                self.controller(self.model, self.data)
                mj.mj_step(self.model, self.data)
                time.sleep(0.002)
            # if not itr % 1000:
            #     print(self.qpos_cmd,self.qvel_cmd)
                itr+=1
        glfw.terminate()
            

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
    xml_file = os.path.join(dirname + "/model/auboi20/aubo_i20.xml")
    urdf_file = os.path.join(dirname + "/model/auboi20/aubo_i20.urdf")
    sim = MujocoSimulator(xml_file,urdf_file) 
    sim.initSimulator()
    sim.runSimulation()
