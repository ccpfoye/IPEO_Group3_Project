# IPEO_Group3_Project
The Final Project for IPEO 2024, Group 3. 

## Marine Debris Detection

Download the data from [here](https://enacshare.epfl.ch/bY2wS5TcA4CefGks7NtXg) and place it in the `/data` directory.

Your repo should look like:
```
|- data/
  |- classification_dataset/
    |- train/
    |- test/
    |- validation/
    |- README.md
  ```

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



Sentinel-2 bands	Sentinel-2A	Sentinel-2B	
Central wavelength (nm)	Bandwidth (nm)	Central wavelength (nm)	Bandwidth (nm)	Spatial resolution (m)
Band 1 – Coastal aerosol	442.7	21	442.2	21	60
Band 2 – Blue	492.4	66	492.1	66	10
Band 3 – Green	559.8	36	559.0	36	10
Band 4 – Red	664.6	31	664.9	31	10
Band 5 – Vegetation red edge	704.1	15	703.8	16	20
Band 6 – Vegetation red edge	740.5	15	739.1	15	20
Band 7 – Vegetation red edge	782.8	20	779.7	20	20
Band 8 – NIR	832.8	106	832.9	106	10
Band 8A – Narrow NIR	864.7	21	864.0	22	20
Band 9 – Water vapour	945.1	20	943.2	21	60
Band 10 – SWIR – Cirrus	1373.5	31	1376.9	30	60
Band 11 – SWIR	1613.7	91	1610.4	94	20
Band 12 – SWIR	2202.4	175	2185.7	185	20