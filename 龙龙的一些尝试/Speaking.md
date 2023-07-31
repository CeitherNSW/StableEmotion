# 1
I have tried two different segmentation models. The first model, Deeplabv3 with ResNet50 as a backbone, didn’t perform well. I received jagged images as output from this model, which were not satisfactory. It's possible that this happened due to an insufficient training set. So, I modified the code to adjust the output of the model using F.interpolate, which led to a lower Loss.

However, when I changed the predictions to a probability distribution using softmax, the output turned into a set of entirely black pictures.

Finally, I decided to use yolov8-seg to complete the task of background removal. However, the labels provided did not meet the model's requirements, which meant I had to reformat them. This involved splitting the long list into dictionaries, extracting the bounding boxes (bboxes), and then reformatting this information into [x_center, y_center, width, height]. I also normalized the data between 0 and 1.

Following this, I used a GPU server to train the model and make predictions on the validation set.

# 2
Long:
After training, the model will automatically save the last.pt and best.pt. We could choose the best.pt as the model and let the model make predictions and the valid set. The returned labels will be saved in the folder. In some cases the model will make prediction more than once for each picture, so we choose the highest confidence as the final result.
