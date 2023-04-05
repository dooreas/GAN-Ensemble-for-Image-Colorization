These are the files for the baseline models trained in the LAB color space.
----------------------------------
The LAB_initial_baseline folder contains the baseline model trained for LAB without early stopping. The main file is:
-Baseline.py
To execute it it requires setting the paths to the correct datset folder first. It will be likely that other paths in the utils.py file will need to be changed in order to create a folder which saves the fotos generated during training. 
-----------------------------------
The LAB_baseline_early folder contains the baseline model trained on LAB with early stopping. The main file is:
-Baseline_with_early_stopping.py
and the especifications to use it are the same as in the above. 

The file:
-Baseline_test_fotos_SSIM.py

is implemented to observe the performance of our model in the test set. It is required to have a resnet34 model .pth file to use it. The reason is that although in this model we do not use a per-category trained colorization model, we realized that the order imposed by the classification of the resnet model in terms of the names of the jpg files was beneficial for us to easily find the test images corresponding to the test images produced by our ensemble. (NEVERTHELESS: see that there is no actual ensemble being implemented in it)
