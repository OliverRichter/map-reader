# MapReader

## About

Tensorflow implementation of the map reading algorithm described in ‘Teaching a Machine to Read Maps with Deep Reinforcement Learning’. This folder includes a copy of the DeepMind lab (https://github.com/deepmind/lab) in which adjustments were made such that the environment is suited for the problem specified in the paper. We started our implementation from a reimplementation of the UNREAL agent (https://arxiv.org/pdf/1611.05397.pdf) done by miyosuda (https://github.com/miyosuda/unreal). Code fragments from this implementation remain in our code.

## Requirements

- TensorFlow
- numpy
- cv2
- pygame
- matplotlib

## How to train
Download the lab folder, install the dependencies required and run the following comment from the lab directory on your system:

$ bazel run //mapreader:train --define headless=osmesa

## How to display results

To view the agent's interactions with the environment, run the following comment from the lab subdirectory:

$ bazel run //mapreader:display --define headless=osmesa

The ‘mapreader_maps’ directory includes all training and test maps described in the paper. To display specific maps, replace the two folders “maps” and “dmlab_level_data” in the directory ‘lab\assets’ with the corresponding folders from ‘mapreader_maps’. Note that the compilation of new maps can take quite some time. To display a specific map, replace the two numbers in the line

DISPLAY_LEVEL = [120,0]

in the file ‘lab\mapreader\constants.py’ with the 2 numbers of the map you’d like to display (given in the map name). Then run the display command above.

## How to load the trained agent
You can find the checkpoints of the trained agent which was used for the evaluation in the paper in the folder ‘trained_agent’. To display the trained agent, switch to the branch 'version_paper', copy the folder ‘mapreader_checkoints’ into your ‘/tmp’ directory and run the display command above.


