# CARLA Dataset

We used synthetic data from CARLA simulator in the experiment VI.A. You need to have the simulator installed, please see the official guide [here](https://carla.readthedocs.io/en/latest/start_quickstart/). It is then possible to extract the data using the prepared python script:
```
sh /path/to/carla/CarlaUE4.sh -RenderOffScreen &
sleep 10s && python3 carla_one_sequence.py [spawn_point] /path/to/output/dir
``` 
There are 155 spawn points on the map (from 0 to 154), which results in 155 sequences of 200 frames. There is also a bash script that generates all these sequences into the current directory:
```
sh carla_full_dataset.sh
```

In each sequence, there are 200 *.mat files (contains pointclouds) and 200 *.jpg images. Please see the `carla_projection` scripts in the repsective inference directories to better understand, how to work with the data structure.

If you use the data, please cite the original research below.

Dosovitskiy, A., Ros, G., Codevilla, F., Lopez, A., & Koltun, V. (2017). CARLA: An open urban driving simulator. In Proceedings of the 1st Annual Conference on Robot Learning (pp. 1-16).

```
@inproceedings{Dosovitskiy17,
  title = { {CARLA}: {An} Open Urban Driving Simulator},
  author = {Alexey Dosovitskiy and German Ros and Felipe Codevilla and Antonio Lopez and Vladlen Koltun},
  booktitle = {Proceedings of the 1st Annual Conference on Robot Learning},
  pages = {1--16},
  year = {2017}
}
```