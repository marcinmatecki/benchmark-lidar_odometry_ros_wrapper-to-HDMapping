# lidar_odometry_ros-converter

## Example Dataset: 

Download the dataset from [Bunker DVI Dataset](https://charleshamesse.github.io/bunker-dvi-dataset/)  

## Intended use 

This small toolset allows to integrate SLAM solution provided by [lidar_odometry_ros_wrapper](https://github.com/93won/lidar_odometry_ros_wrapper) with [HDMapping](https://github.com/MapsHD/HDMapping).
This repository contains ROS 2 workspace that :
  - submodule to tested revision of lidar_odometry_ros_wrapper
  - a converter that listens to topics advertised from odometry node and save data in format compatible with HDMapping.

## Building

Clone the repo
```shell
mkdir -p /test_ws/src
cd /test_ws/src
git clone https://github.com/marcinmatecki/lidar_odometry_ros_wrapper-to-HDMapping.git --recursive
cd ..
colcon build
```

## Usage - data SLAM:

Prepare recorded bag with estimated odometry:

In first terminal record bag:
```shell
ros2 bag record /odometry /feature_points
```

and start odometry:
```shell 
cd /test_ws/
source ./install/setup.sh # adjust to used shell
ros2 launch lidar_odometry_ros lidar_odometry.launch.py config_file:=<config_path> use_sim_time:=true pointcloud_topic:=<topic>
ros2 bag play {path_to_bag}
```

## Usage - conversion:

```shell
cd /test_ws/
source ./install/setup.sh # adjust to used shell
ros2 run lidar-odometry-ros-to-hdmapping listener <recorded_bag> <output_dir>
```

## Convert(If it's a ROS1 .bag file):

```shell
rosbags-convert --src {your_downloaded_bag} --dst {desired_destination_for_the_converted_bag}
```

## Record the bag file:

```shell
ros2 bag record /odometry /feature_points -o {your_directory_for_the_recorded_bag}
```

## lidar_odometry_ros Launch:

```shell
cd /test_ws/
source ./install/setup.sh # adjust to used shell
ros2 launch lidar_odometry_ros lidar_odometry.launch.py config_file:=<config_path> use_sim_time:=true pointcloud_topic:=<topic>
ros2 bag play {path_to_bag}
```

## During the record (if you want to stop recording earlier) / after finishing the bag:

```shell
In the terminal where the ros record is, interrupt the recording by CTRL+C
Do it also in ros launch terminal by CTRL+C.
```

## Usage - Conversion (ROS bag to HDMapping, after recording stops):

```shell
cd /test_ws/
source ./install/setup.sh # adjust to used shell
ros2 run lidar-odometry-ros-to-hdmapping <recorded_bag> <output_dir>
```
