from __future__ import print_function
  
import numpy as np
from numpy.linalg import norm, solve,pinv,inv
  
import pinocchio
class Kinematics:
    def __init__(self, urdf_path: str):
        """Initializes the kinematics solver with a robot model."""
        self.urdf_path = urdf_path
        self.model =pinocchio.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()
        self.joint_id = self.model.getJointId("joint5") 
        self.frame_id = self.model.getFrameId("wrist3_Link")
        self.eps = 1e-4
        self.IT_MAX = 500
        self.DT = 1e-1
        self.damp = 1e-8
        self.kErrorLimt = 2e-1

    ##################################
    #pos_des:cartesian position in world frame
    #rotm_des:rotation matrix from joint frame to world frame
    #vel_des: velocity in world frame
    #omega_des: angular velocity in word frame
    ##################
    def ikine(self,qref,pos_des,rotm_des,vel_des,omega_des,qpos_upper_limit,qpos_lower_limit):
        q = pinocchio.utils.zero(self.model.nq)
        for i in range(self.model.nq):
            q[i]=qref[i]
        #get target joint positions
        oMdes = pinocchio.SE3(rotm_des, pos_des) 
        itr=0
        success = False
        while True:
            pinocchio.forwardKinematics(self.model,self.data,q)
            dMi = self.data.oMi[self.joint_id].actInv(oMdes)
            err = pinocchio.log(dMi).vector
            if norm(err) < self.eps:
                success = True
                break
            if itr >= self.IT_MAX:
                success = False
                print("IK failed")
                break
            # if norm(err) > self.kErrorLimt:
            #     err=err/norm(err)*self.kErrorLimt
            J = pinocchio.computeJointJacobian(self.model,self.data,q,self.joint_id)
            J = -np.dot(pinocchio.Jlog6(dMi.inverse()), J)
            v = - J.T.dot(solve(J.dot(J.T) + self.damp * np.eye(6), err))
            # print(q)
            # print("v",v*self.DT)
            q = pinocchio.integrate(self.model,q,v*self.DT)
           
            #Limit joint positions
            # for i in range(self.model.nq):
            #     if q[i] > qpos_upper_limit[i]:
            #         q[i] = qpos_upper_limit[i]
            #     if q[i] < qpos_lower_limit[i]:
            #         q[i] = qpos_lower_limit[i]

            # if not i % 100:
            #     print('%d: error = %s' % (i, err.T))
            itr += 1
        #ge target joint velocities
        if success:
                print("Convergence achieved!")
        else:
                print("\nWarning: the iterative algorithm has not reached convergence to the desired precision")
  
        rotm=self.data.oMi[self.joint_id].rotation
        vel_des_body=np.zeros(6)
        vel_des_body[0:3]=rotm.T@np.array([vel_des[0],vel_des[1],vel_des[2]])
        vel_des_body[3:6]=rotm.T@np.array([omega_des[0],omega_des[1],omega_des[2]])
        J = pinocchio.computeJointJacobian(self.model,self.data,q,self.joint_id)
        dq=solve(J,vel_des_body)
        return q,dq
    
    def fkine(self,qpos):
        q = pinocchio.utils.zero(self.model.nq)
        for i in range(self.model.nq):
            q[i]=qpos[i]
        pinocchio.forwardKinematics(self.model,self.data,q)
        return self.data.oMi[self.joint_id].translation,self.data.oMi[self.joint_id].rotation
