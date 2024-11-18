# IPEO_Group3_Project
The Final Project for IPEO 2024, Group 3. 

## Marine Debris Detection

Download the data from [here](https://enacshare.epfl.ch/bY2wS5TcA4CefGks7NtXg) and place it in the `/data` directory.

Here is the overview from the project overview:

### Marine Debris Detection

#### Task 
Ocean plastic pollution is a threat for marine and coastal ecosystems and a major societal and environmental concern. Being able to monitor it with a timely detection is of big importance to coordinate cleaning efforts. Floating marine debris tends to create filaments referred to as windrows, generally used as proxies for plastic pollution. They can reach more than 50m of width and 500m of length, making them visible from optical satellite sensors such as Sentinel-2, having a resolution of 10 to 20m. The goal of the project is to develop an classifier that identifies the presence of floating marine debris in Sentinel-2 images.
#### Data 
Students will use a dataset composed of 12-bands Sentinel-2 image patches of size 32×32 pixels. Each image is associated with a binary label: ”0” for negative labels (absence of floating debris), ”1” for positive labels (presence of floating debris). The dataset is composed of 53535 training patches, 7436 validation patches and 13386 test patches. The samples for the validation and test set have been sampled to contain an equal number of positive and negative examples.

#### Challenges
• The task is an image classification task, where the challenge is to associate a label to each image.
You will have to study the performance of the classifier in three different scenarios: 
- [ ] with the use of all the 12 spectral bands,
- [ ] with the use of only RGB+NIR bands,
- [ ] with the use of only the visible spectrum,
to study the sensitivity of the model to the different spectral bands.
• The dataset is strongly imbalanced with only few positive examples in the training set: this needs
to be taken into account for an effective training of the model.
• The dataset is obtained from a segmentation dataset with per-pixel annotations, such that if at least
one positive pixel is present in the image, the image class is set to ”1”. Thus, for some images, it
can be challenging to correctly classify them as only few pixels contain floating material.
