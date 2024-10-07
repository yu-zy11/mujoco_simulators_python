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
    plt.ylabel('x pos /m')
    plt.subplot(3,2,3)
    plt.plot(time_array,pos_array[1],label='y pos')
    plt.xlabel('time /s')
    plt.ylabel('y pos /m')
    plt.subplot(3,2,5)
    plt.plot(time_array,pos_array[2],label='z pos')
    plt.xlabel('time /s')
    plt.ylabel('z pos /m')
    plt.subplot(3,2,2)
    plt.plot(time_array,vel_array[0],label='x vel')
    plt.xlabel('time /s')
    plt.ylabel('x vel /m/s')
    plt.subplot(3,2,4)
    plt.plot(time_array,vel_array[1],label='y vel')
    plt.xlabel('time /s')
    plt.ylabel('y vel /m/s')
    plt.subplot(3,2,6)
    plt.plot(time_array,vel_array[2],label='z vel')
    plt.xlabel('time /s')
    plt.ylabel('z vel /m/s')
    plt.legend()
    plt.show()


    
#################### test
if __name__ == '__main__':
    max_vel=1.0
    max_acc=2.0
    #设置目标点的位置
    pos_array=np.array([[-280,  336,   673, 673, 1739, 1739, 1739,  2674, 2674, 3403, 3628, 3628, 3628],
                        [-221, -221,  -147,-147,-147, -147, -147,  -147, -147, -147, -388, -344, -344],
                        [834.7 ,834.7, 700, 453, 807,  453,  117.6, 807,  658,  410,  1352, 954,  477]]).dot(0.001)
    start_pos=pos_array[:,0]
    end_pos=pos_array[:,1]
    duration_array=np.zeros(pos_array.shape[1]-1)
    step=0.001
    for i in range(pos_array.shape[1]-1):
        start_pos=pos_array[:,i]
        end_pos=pos_array[:,i+1]
        pos,vel,duration=trapezoid_planning(start_pos,end_pos,max_vel,max_acc,0)
        duration_array[i]=duration
    print("duration_array",duration_array)
    
    # example for plotting
    num_point=5000
    time_plot=np.zeros(num_point)
    pos_plot=np.zeros((3,num_point))
    vel_plot=np.zeros((3,num_point))
    step=0.001
    start_pos=pos_array[:,0]
    end_pos=pos_array[:,1]
    for i in range(num_point):
        time=i*step
        pos,vel,duration=trapezoid_planning(start_pos,end_pos,max_vel,max_acc,time)
        time_plot[i]=time
        pos_plot[:,i]=pos.copy()
        vel_plot[:,i]=vel.copy()
    PlotPositionAndVelocity(time_plot,pos_plot,vel_plot)