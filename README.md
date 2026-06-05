# Navigation System for ROS2 Robot using Intel Realsense Camera

## Mapping.py
Allows user to move robot around in an enviroment and map coodinates through wheel encoders using the Odometry messages from ros2 to produce a text file pf all recorded positons.
"Landmarks" (represented by ArUco codes) are recorded in the terminal, providing an id of the code and the x,y,z in relation to the robots origin.

## Navigating.py
Uses a tuple of x and y coodinates to move robot from one postion to another. Trajectores were calculated using a reward field planner made by my supervisor for this project
to find the best path.

## Disclamers
Code did not work on robot due to wheel slipage but might be affected by the code as pure pursuit is a simple algorithm for this kind of project, therfore a fuzzy logic
or PID approach might be better suited to reduce this error. I am very new to robotics and was not taught any during my 3 years of my Computer Science Undergraduate until this capstone project
so the answer might be more obvious to those with expreiance but this is what I came up with within a year of working on this.
