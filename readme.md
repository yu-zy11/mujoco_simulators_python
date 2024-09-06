open tenimal and cd mujoco_simulator
```
python3 ./mujoco_simulator_biped.py 
```


if you want to use lcm on single host,and do not connect to network, open ternimal and run:
  sudo ifconfig lo multicast
  sudo route add -net 224.0.0.0 netmask 240.0.0.0 dev lo