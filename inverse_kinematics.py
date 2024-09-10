from __future__ import print_function
  
import numpy as np
from numpy.linalg import norm, solve,pinv,inv
  
import pinocchio
class pinocchio_kinematics:
    def __init__(self, urdf_path: str):
        """Initializes the kinematics solver with a robot model."""
        self.urdf_path = urdf_path
        self.model =pinocchio.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()
        self.joint_id = self.model.getJointId("joint5") 
        self.frame_id = self.model.getFrameId("wrist3_Link")
        self.eps = 1e-4
        self.IT_MAX = 1000
        self.DT = 1e-1
        self.damp = 1e-12
        self.max_error = 0.2

    ##################################
    # pos_des:cartesian position in world frame
    #rotm_des:rotation matrix from joint frame to world frame
    #vel_des: velocity in world frame
    #omega_des: angular velocity in word frame
    ##################
    def ikine(self,qref,pos_des,rotm_des,vel_des,omega_des):
        q = pinocchio.utils.zero(self.model.nq)
        for i in range(self.model.nq):
            q[i]=qref[i]
        #get target joint positions
        oMdes = pinocchio.SE3(rotm_des, pos_des) 
        i=0
        jacobian=np.eye(6)
        while True:
            pinocchio.forwardKinematics(self.model,self.data,q)
            dMi = oMdes.actInv(self.data.oMi[self.joint_id])
            err = pinocchio.log(dMi).vector
            if norm(err) < self.eps:
                success = True
                break
            if i >= self.IT_MAX:
                success = False
                break
            # simpliest line search method to avoid overshoot
            if norm(err)>self.max_error:
                err = err/norm(err)*self.max_error
            
            J = pinocchio.computeJointJacobian(self.model,self.data,q,self.joint_id)
            v = - J.T.dot(solve(J.dot(J.T) + self.damp * np.eye(6), err))
            q = pinocchio.integrate(self.model,q,v*self.DT)
            # if not i % 10:
            #     print('%d: error = %s' % (i, err.T))
            i += 1
            jacobian=J
        #ge target joint velocities
        rotm=self.data.oMi[self.joint_id].rotation
        vel_des_body=np.zeros(6)
        vel_des_body[0:3]=rotm.T@np.array([vel_des[0],vel_des[1],vel_des[2]])
        vel_des_body[3:6]=rotm.T@np.array([omega_des[0],omega_des[1],omega_des[2]])
        dq=solve(jacobian,vel_des_body)
        return q,dq
    
    def fkine(self,qpos):
        q = pinocchio.utils.zero(self.model.nq)
        for i in range(self.model.nq):
            q[i]=qpos[i]
        pinocchio.forwardKinematics(self.model,self.data,q)
        return self.data.oMi[self.joint_id].translation,self.data.oMi[self.joint_id].rotation
