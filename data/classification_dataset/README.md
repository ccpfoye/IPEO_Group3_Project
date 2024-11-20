# Marine Debris dataset for image classification
The training, validation and test subsets contain image patches of size $32\times 32$ pixels. Each image patch contains 12 spectral bands and is acquired with Sentinel-2. Thus, the band order follows the Sentinel-2 convention.

Each image patch is associated with a binary label:
- 0: absence of floating debris
- 1: presence of floating debris

The dataset is constructed to train classification models on it to detect the presence of marine debris on the images.

An example of code snippet to read the images is provided below:
```python
import os
import pickle

savedir_test = "test"

test_images = []
test_labels = []
for filename in os.listdir(savedir_test):
    with open(os.path.join(savedir_test, filename), 'rb') as f:
        image_label_tuple = pickle.load(f)
        test_images.append(image_label_tuple[0])
        test_labels.append(image_label_tuple[1])
```