#!/bin/bash
for i in $(seq 0 1 154)
do
sh /path/to/carla/CarlaUE4.sh -RenderOffScreen &

sleep 10s && python3 carla_one_sequence.py "$i" "./"

x=`ps -A | grep Carla | cut -d " " -f3`

kill -9 $x
done