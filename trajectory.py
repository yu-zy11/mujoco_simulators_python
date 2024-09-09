import numpy as np
import matplotlib.pyplot as plt
def trapezoid_planning(start_pos,end_pos,max_vel, max_acc, time):
    #计算距离和方向
    ZERO=0.0000001
    dist = np.linalg.norm(start_pos - end_pos)
    if(dist < ZERO):
        return end_pos, np.zeros(3),0  #pos vel
    direction = (end_pos - start_pos) / dist
    key_time=max_vel/max_acc
    d1=0.5*max_acc*key_time**2
    #
    if dist < 2*d1:#无匀速区间
        t1 =np.sqrt(dist / max_acc)
        t2=t1
        if time<t1:
            pos = start_pos + 0.5 * max_acc * time * time * direction
            vel = max_acc * time* direction
        elif time < t1 + t2:
            pos=start_pos+(0.5*dist+max_acc*t1*(time-t1)-0.5*max_acc*(time-t1)**2)*direction
            vel=(max_acc*t1-max_acc*(time-t1))* direction
        else:
            pos = end_pos
            vel = np.zeros(3)
        duration=t1+t2
        return pos, vel,duration
    else:#有匀速区间
        t1 = max_vel / max_acc
        t3 = t1
        t2 = (dist - 2 * d1) / max_vel
        if(time < t1):
            pos = start_pos + 0.5 * max_acc * time * time * direction
            vel = max_acc * time* direction
        elif(time < t1 + t2):
            pos = start_pos + (d1 + max_vel * (time - t1))* direction
            vel = max_vel* direction
        elif (time < t1 + t2 + t3):
            pos = start_pos + (dist - 0.5*max_acc * (t1 + t2 + t3-time)**2) * direction
            vel = (max_vel-max_acc*(time-t1-t2))* direction
        else:
            pos = end_pos
            vel = np.zeros(3)
        duration=t1+t2+t3
        return pos, vel,duration

def PlotPositionAndVelocity(time_array,pos_array,vel_array):
    plt.subplot(3,2,1)
    plt.plot(time_array,pos_array[0],label='x pos')
    plt.xlabel('time /s')
    plt.ylabel('pos /m')
    plt.subplot(3,2,3)
    plt.plot(time_array,pos_array[1],label='y pos')
    plt.xlabel('time /s')
    plt.ylabel('pos /m')
    plt.subplot(3,2,5)
    plt.plot(time_array,pos_array[2],label='z pos')
    plt.xlabel('time /s')
    plt.ylabel('pos /m')
    plt.subplot(3,2,2)
    plt.plot(time_array,vel_array[0],label='x vel')
    plt.xlabel('time /s')
    plt.ylabel('vel /m/s')
    plt.subplot(3,2,4)
    plt.plot(time_array,vel_array[1],label='y vel')
    plt.xlabel('time /s')
    plt.ylabel('vel /m/s')
    plt.subplot(3,2,6)
    plt.plot(time_array,vel_array[2],label='z vel')
    plt.xlabel('time /s')
    plt.ylabel('vel /m/s')
    plt.legend()
    plt.show()


    
#################### test
if __name__ == '__main__':
    start_pos=np.array([0,0,0])
    end_pos=np.array([1,1,1])
    max_vel=1
    max_acc=1
    num_point=5000
    time_array=np.zeros(num_point)
    pos_array=np.zeros((3,num_point))
    vel_array=np.zeros((3,num_point))
    step=0.001
    for i in range(num_point):
        time=i*step
        pos,vel=trapezoid_planning(start_pos,end_pos,max_vel,max_acc,time)
        time_array[i]=time
        pos_array[:,i]=pos.copy()
        vel_array[:,i]=vel.copy()
    PlotPositionAndVelocity(time_array,pos_array,vel_array)