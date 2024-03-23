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

model.compile(
    optimizer=keras.optimizers.Adam(1e-4), loss = "sparse_categorical_crossentropy"
)

# TODO: Handle tensorboard callback in a better way. Use automatic folder naming.
callbacks = [
    keras.callbacks.ModelCheckpoint("oxford_segmentation.tf", save_best_only=True),
    TensorBoard(log_dir='logs/2',histogram_freq=1,write_images=True)
]

model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=valid_dataset,
    callbacks=callbacks,
    verbose=2,
)

# Test the model
test_dataset = mod.get_dataset(batch_size, img_size,
                               test_input_img_paths, test_target_img_paths)

# Open the best persisted model
model = keras.models.load_model("oxford_segmentation.tf")

# Predict on the test dataset
predictions = model.predict(test_dataset)

# TODO: Implement mehtod to calculate ACC, AUROC, IoU, and Dice for the predictions
acc, auroc, iou, dice = mod.calculate_iou_and_dice(test_dataset, predictions)

# Print the results
print(f"ACC: {acc}")
print(f"AUROC: {auroc}")
print(f"IoU: {iou}")
print(f"Dice: {dice}")

#Visualize the predictions for the first 4 images
# TODO: Use visualization methods you prepared of the predictions. Save the visualizations in 4 files.

