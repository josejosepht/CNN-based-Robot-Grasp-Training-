# CNN-based-Robot-Grasp-Training-
This project involves the development of a grasp prediction system using a convolutional neural network (CNN) trained on a dataset of top-down RGB images. The project includes a grasp simulator where a Panda Arm attempts grasps on randomly placed objects. The CNN architecture, MobileUNet, is implemented to output a distribution over pixel location and gripper rotation for successful grasps. The network is trained and evaluated on CPU, with the option to utilize a GPU if available. Data augmentation is introduced to account for object rotations during training. The project aims to predict successful grasps accurately, taking into account the challenges of object placement and orientation, and provides a comprehensive solution for grasp prediction using a CNN.

The different scripts implemented are as follows:
*  `simulator.py` : The simulator contains a Panda Arm and a square workspace where small objects are randomly placed. After every reset, the robot arm attempts a grasp with the gripper aligned along x- or y-axis. An RGB camera (not visible) is positioned to view the workspace from above, and the camera feed is shown in the top left panel:

![image](https://github.com/josejosepht/CNN-based-Robot-Grasp-Training-/assets/97187460/795ee9df-ae14-4847-b250-9ae6a5b28d14)

The simulator from above was used to collect a dataset of 1,000 successful grasps, split into a train(`train_dataset.npz`) and validation set(`val_dataset.npz`). For each grasp, a 64 × 64 pixel RGB image is captured of the scene and we record the pixel location (px, py) and rotation (0 or 90 degrees) of the gripper.
The simulation involves tasks such as resetting the environment, rendering top-down RGB images of the scene, executing grasps at specific pixel locations, and determining the success or failure of the grasps based on the resulting object positions. The code also includes utilities for object manipulation, joint control, camera rendering, and workspace visualization. Overall, the code aims to provide a realistic grasp simulation environment for training and testing grasp prediction algorithms.


*  `trainer.py` : Implements the training and evaluation pipeline for a MobileUNet model(refer below architecture inspired by Fig. 10 of "Searching for MobileNetV3"
(https://arxiv.org/pdf/1905.02244.pdf). Intermediate tensor shapes are labeled without batch dimension) used for predicting the success of grasps given top-down RGB images of a scene:

![image](https://github.com/josejosepht/CNN-based-Robot-Grasp-Training-/assets/97187460/8bf783c7-faa5-4823-ab71-5fee8e366784)


The MobileUNet model is built using a pretrained MobileNetv3 backbone(`grasp_mobilenet.pt`) and consists of additional convolutional layers and upsampling operations. The code includes functionalities for loading the grasp dataset, creating data loaders, defining the model architecture, training the model using cross-entropy loss, saving and loading model weights, and plotting the learning curves and prediction results. The main function orchestrates the training process and saves the best model checkpoint based on validation loss.
The network will output an image with the same size as the input image, where each pixel has two channels corresponding to the two different gripper rotations (0 degrees and 90 degrees about the z-axis). In other words, the network outputs a distribution over pixel location and gripper rotation for a successful grasp.


`trainer.py`, while training, saves the loss curves as 'loss_curves.png' and predictions as 'predictions.png':

![loss_curves](https://github.com/josejosepht/CNN-based-Robot-Grasp-Training-/assets/97187460/74ff30c2-6fcb-4e8a-9334-df89b42295e2)

![predictions](https://github.com/josejosepht/CNN-based-Robot-Grasp-Training-/assets/97187460/7e7722c6-070e-4342-9e64-ac2c0d87202a)

* `evaluate_model.py` : An evaluation script that uses a trained MobileUNet model to perform grasp prediction on simulated images. It takes a saved model file path (`grasp_mobilenet.pt`) as input and evaluates the model's performance by executing 50 grasps in a simulated environment. The script initializes a simulator and loads the MobileUNet model. It then iterates a specified number of times, rendering images from the simulator, converting them to tensors, and using the model to predict the grasp's success. The success rate of the grasps is recorded and displayed during the evaluation process.

* `dataset.py` : Defines a custom dataset class called GraspDataset. This dataset is designed to store and provide access to a collection of successful grasp data points. Each data point in the dataset consists of a top-down RGB image of a scene with dimensions 64x64 and a corresponding grasp pose represented by the gripper position in pixel space and rotation (0 degrees or 90 degrees). The network is re-trained with rotation data augmentations and obtained a 44% success rate as compared to the previous 30 % averaged over the 50 grasp attempts by running `evaluate_model.py` again

![image](https://github.com/josejosepht/CNN-based-Robot-Grasp-Training-/assets/97187460/ed3955ab-56a9-433d-ae77-e7796f79db42)

![image](https://github.com/josejosepht/CNN-based-Robot-Grasp-Training-/assets/97187460/65e94e63-9a44-44d4-b12e-2828b94315a9)

# Interesting Inferences and future work
* Simulators provide faster experimentation and enhanced safety by simulating a large number of iterations without physical limitations or handling unsafe objects, but may lack accurate modeling of physical parameters and exhibit visual gaps in grasp iterations.

* The network's architecture incorporates both high-resolution and low-resolution pathways to capture local details (e.g., object shape, gripper geometry) and global contextual information (e.g., object location, relationships with other objects), respectively, enabling a comprehensive understanding of the grasping task.

* The validation loss curve during training can be noisy due to factors like smaller validation set size and limited diversity, but it remains a crucial metric for evaluating the model's generalization ability.

* To improve generalization, the network can be trained with data augmentation techniques such as random crops, flips, rotations, and gripper orientation variations, effectively increasing the diversity of training data and helping the network adapt to different translations and orientations of objects in the scene.
