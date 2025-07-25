This is a feature for testing purposes only.

If we want to generate sequence-level data, we should 
1. download the kitti tracking dataset, and organize them within kitti detection format.
2. manually specify the position and orientation of each frame for object insertion.
3. modify the code in demo.py, so that we can insert object with a specified pose. We provide a serious of poses in ./poses.
4. run python demo.py and generate data.
