# 1
I experimented with two segmentation models. The first one, Deeplabv3 with ResNet50, didn't yield satisfactory results, producing jagged images. I suspect this was due to an inadequate training set. By modifying the code using F.interpolate, I lowered the model's Loss. But, the output became completely black when I shifted predictions to a softmax probability distribution.

I finally opted for yolov8-seg for background removal. I had to reformat the labels to fit the model's needs, splitting the long list into dictionaries, and changing the bounding box information to a normalized format. I trained the model on a GPU server and used it to make predictions on a validation set.

# 2
Post-training, the model auto-saves as last.pt and best.pt. We typically use best.pt for predictions on the valid set. Predictions might be made multiple times per image, so we choose the one with the highest confidence as the final result.