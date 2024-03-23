import keras
import model as mod
from tensorflow.keras.callbacks import TensorBoard

batch_size = 32
epochs = 50
img_size = (256, 256)

train_input_img_paths = 'data/train-image-tiles/'
train_target_img_paths = 'data/train-mask-tiles/'

val_input_img_paths = 'data/valid-image-tiles/'
val_target_img_paths = 'data/valid-mask-tiles/'

test_input_img_paths = 'data/test-image-tiles/'
test_target_img_paths = 'data/test-mask-tiles/'

train_dataset = mod.get_dataset(batch_size, img_size,
                                train_input_img_paths, train_target_img_paths)

valid_dataset = mod.get_dataset(batch_size, img_size,
                                val_input_img_paths, val_target_img_paths)

model = mod.get_model()

optimizer = keras.optimizers.Adam(1e-8)
mod.compile_model(model, optimizer=optimizer)

# TODO: Handle tensorboard callback in a better way. Use automatic folder naming.
callbacks = mod.make_callbacks()

mod.train_model(model, train_dataset=train_dataset, validation_dataset=valid_dataset, epochs=epochs, callbacks=callbacks, verbose=2)

# Test the model
test_dataset = mod.get_dataset(batch_size, img_size,
                               test_input_img_paths, test_target_img_paths)

# Open the best persisted model
model = mod.load_model()

# Predict on the test dataset
predictions = mod.model_prediction(model, valid_dataset)

# TODO: Implement mehtod to calculate ACC, AUROC, IoU, and Dice for the predictions
acc, auroc, iou, dice = mod.calculate_iou_and_dice(test_dataset, predictions)

# Print the results
print(f"ACC: {acc}")
print(f"AUROC: {auroc}")
print(f"IoU: {iou}")
print(f"Dice: {dice}")

#Visualize the predictions for the first 4 images
# TODO: Use visualization methods you prepared of the predictions. Save the visualizations in 4 files.

