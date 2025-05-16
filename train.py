import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json

print("Available GPU:", tf.config.list_physical_devices('GPU'))


IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_INITIAL = 10
EPOCHS_FINE_TUNE = 5
AUTOTUNE = tf.data.AUTOTUNE
LEARNING_RATE = 1e-3
FINE_TUNE_LR = 1e-5


train_dir = "/content/Plant_Deasease_Detection/train"
valid_dir = "/content/Plant_Deasease_Detection/valid"


train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)
class_names = train_ds_raw.class_names
with open("/content/class_names.json", "w") as f:
    json.dump(class_names, f)


train_ds = train_ds_raw.prefetch(buffer_size=AUTOTUNE)





valid_ds = tf.keras.utils.image_dataset_from_directory(
    valid_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
).prefetch(buffer_size=AUTOTUNE)


data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
], name="data_augmentation")


base_model = EfficientNetB0(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet')
base_model.trainable = False

inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
outputs = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs, outputs)

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])


callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ReduceLROnPlateau(patience=2, factor=0.2)
]


print("Phase 1 Training")
model.fit(train_ds, validation_data=valid_ds, epochs=EPOCHS_INITIAL, callbacks=callbacks)

base_model.trainable = True
for layer in base_model.layers[:150]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=FINE_TUNE_LR), loss='categorical_crossentropy', metrics=['accuracy'])

print("Phase 2 Fine-Tuning")
model.fit(train_ds, validation_data=valid_ds, epochs=EPOCHS_FINE_TUNE, callbacks=callbacks)


model.save("/content/plant_disease_model.h5")
print("Model saved.")
