# VotingNet6DPose
This is a 6D pose network based on PVNet



# Code Structure

* configs - Some setting are configured here
* datasets - LINEMOD dataset
* evaluator - The evaluation process is implemented here
* log_info - Used for storing models and results
* nets - The network model is implemented here
* trainer - The supplementary class for training the network
* utils
  * decorators - Helper decorators
  * io_utils - IO Helper
  * refinement - ICP refinement process is here
  * visual_utils - Visualization helper



# Usage

There are some steps you need to finish before running the program.

* Specify the *path of the dataset* in your own machine in **configs/constants.py**
* 

