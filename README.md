# Project: Application of PCA in Image Processing Using the Yale Faces Dataset
## Objective:
To implement Principal Component Analysis (PCA) on the Yale Faces dataset and evaluate its effectiveness in facial recognition by classifying images based on Euclidean distance and PCA-derived representative images.

## Dataset Description:
The Yale Faces dataset consists of images of 15 subjects, each with six different grayscale images captured under varying facial expressions. Each image has a resolution of 243 x 320 pixels.

## Methodology:
### Image Vectorization:

Convert each image into a column vector of size 77760 x 1 by stacking the columns of the digital image one below the other (vec operation on a matrix).

### Facial Recognition Using Euclidean Distance:

Use the normal images of each subject as the standard.
Compute the Euclidean distance between each test image and the standard images.
Classify the images based on the minimum Euclidean distance.
Evaluate the classification accuracy.

### Principal Component Analysis (PCA) for Image Representation:

Perform PCA on the six images of each subject.
Determine the first principal component (PC1) for each subject.
Use PC1 as the weighted combination of the six images that best represents the subject.
Store these representative images in a database.
Classify the original images by comparing them to the representative images in the database using Euclidean distance.
Evaluate the classification accuracy.

### Improving Classification with Multiple Representative Images:

Instead of storing only one representative image, store the first two principal components (PC1 and PC2) for each subject.
Use these two representative images to classify the original images.
Evaluate the classification accuracy based on these two representative images.
