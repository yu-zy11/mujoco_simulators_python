import pinocchio
import os
import numpy as np
import sys

class PinocchioInterface():
    def __init__(self, urdf_path: str):
        self.urdf_path = urdf_path
        self.model = pinocchio.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()
        self.frame_id=0

    def setEndEffector(self,end_effector_name:str):
        self.frame_id = self.GetFrameId(end_effector_name)

    def forwardKinematics(self,qpos):
        pinocchio.forwardKinematics(self.model,self.data,qpos)
        return self.data.oMi
    def GetJacobian(self,qpos,):
        Jacobin = pinocchio.computeJointJacobian(self.model,self.data,qpos,self.joint_id)
        print("Jacobin",Jacobin)
        return self.data.J

    def GetJointId(self,joint_name:str)->int:
        if self.model.existJointName(joint_name):
            return self.model.getJointId(joint_name)
        else:
            print(joint_name+" not exist in urdf")
            file=sys.stderr
            sys.exit(1)
            return -1
    
    def GetFrameId(self,link_name:str)->int:
        if self.model.existBodyName(link_name):
            return self.model.getBodyId(link_name)
        else:
            print(link_name+" not exist in urdf")
            file=sys.stderr
            sys.exit(1)
            return -1

    def UpdateKinematics(self,qpos:np.ndarray):
        if qpos.shape[0] != self.model.nq:
            print("qpos length is not equal to model.nq")
            file=sys.stderr
            sys.exit(1)
        pinocchio.forwardKinematics(self.model,self.data,qpos)
        pinocchio.updateFramePlacements(self.model,self.data)
        pinocchio.computeJointJacobians(self.model,self.data)
        
    
    def GetFrameTraslation(self,frame_id:int):
        return np.array(self.data.oMf[frame_id].translation)

    def GetFrameRotation(self,frame_id:int):
        return np.array(self.data.oMf[frame_id].rotation)

    def GetFrameJacobian(self,frame_id:int):
        return pinocchio.getFrameJacobian(self.model,self.data,frame_id,pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED)


if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    urdf_file = os.path.join(dirname + "/model/auboi20/aubo_i20.urdf")

    pino = PinocchioInterface(urdf_file)
    c=pino.GetJointId("joint5")
    d=pino.GetFrameId("wrist3_Link")
    a=pino.model.getJointId("joint511")
    pino.setEndEffector("wrist3_Link")
    qpos=np.array([0,0,3.1415926/2,0,0,0])
    pino.UpdateKinematics(qpos)
    trans=pino.GetFrameTraslation(pino.frame_id)
    rot=pino.GetFrameRotation(pino.frame_id)
    jac=pino.GetFrameJacobian(pino.frame_id)
    b=1