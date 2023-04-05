These files are for training a per category Pix2Pix model and then to test our ensemble in LAB.
----------------------------
The main files are:
-Training_model_per_cat.py
-Ensemble_FINAL_LAB.py
and they always have to be executed with the other files in the same directory. 
-------------------------
 The file "Training_model_per_cat.py" does a per category training of a Pix2Pix model in the LAB color space with early stopping implemented. It is required to use a batch size of 16 and to set the paths to the dataset correctly (as well as other paths that are used to create folders in which numerical and image data is saved). The dataset has to be organised as:
-category:
	- Train
		-*.jpg
	- Validation
		-*.jpg
The images are in color.
--------------------------
 The file "Ensemble_FINAL_LAB.py" is used to test numerically and give the generated images of our model given the test set. The requirements are the following:
-You need to have a resnet model .pth file saved in the same directory. 
-You need to set the batch size in the "dataset.py" file back to 1.
- You need to have the test set organised simply as:
-test
	-*.jpg
The images are in color. 