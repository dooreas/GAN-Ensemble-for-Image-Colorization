YUV COLOR SPACE:
------------------
The files needed to excute our baseline model with early stopping implemented are:
-Baseline_early_stopping.py (The main one)
-dataset.py
-loss.py
-models.py
-utils.py

The file:

-Baseline_test_fotos_SSIM_YUV.py (Important set bs=1 in the dataset.py file in order to use it)

is implemented to observe the performance of our model in the test set. It is required to have a resnet34 model .pth file to use it. The reason is that although in this model we do not use a per-category trained colorization model, we realized that the order imposed by the classification of the resnet model in terms of the names of the jpg files was beneficial for us to easily find the test images corresponding to the test images produced by our ensemble. (NEVERTHELESS: see that there is no actual ensemble being implemented in it)