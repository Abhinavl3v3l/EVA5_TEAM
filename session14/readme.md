## Team Members
* Abhinav Rana
* Madhu Charan
* Pruthiraj Jaisingh
* Prashanth Shinagare

Colab File - https://colab.research.google.com/drive/11RHX60u2fNkbNQ14T1XWV3xNHmtp3cNA#scrollTo=yGqJiLw1dwVw

## Dataset Preparation for Monocular Depth Estimation

* We have a total of 3590 images of people wearing Hardhat, Vest, Mask and Boots which will be used for Dataset generation.
* We need heat maps, Segmented images for the project
* We used [Intel's Midas](https://github.com/intel-isl/MiDaS) repository to generate the below heat/depth map in gray scale which can be converted to Colored map using Open CV color map but based on a fact that DL model is not biased towards the color we thought first to test with gray scale output and later change it to Colored map.

* Next we used [PlanerCNN](https://github.com/NVlabs/planercnn) code to generate segmented images that looks like below which will be required for next assignment.
* The Depth Image and pLanerCNN output will be as follows -
<img src="https://github.com/pruthiraj/EVA5_TEAM/blob/master/session14/Midas/114.png?raw=true" alt="MIDAS DEPTH MAP">

<img src ="https://github.com/pruthiraj/EVA5_TEAM/blob/master/session14/planercnn/1007_segmentation_0_final.png?raw=true" alt="PlanerCNN Output">

Note: The Planercnn code will also produce depth maps but are not as good as MIDAS one's So we have ommitted them and only considered Segmented image outputs.

## The Notebooks used for the image generation are -

* [MIDAS Depth map Generator Colab]()
* [PlanerCNN segmented image Generator]()
