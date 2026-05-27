import tensorflow as tf
import json

parametros_file = open('dataset/parameters.jsonc', 'r', encoding='utf-8')
parametros = json.load(parametros_file)


def preprocess_health(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    grayscale = tf.image.rgb_to_grayscale(image)
    grayscale_3ch = tf.image.grayscale_to_rgb(grayscale)
    return grayscale_3ch, label


def preprocess_ultimate(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)

    hsv = tf.image.rgb_to_hsv(image)

    h, s, v = tf.split(hsv, 3, axis=-1)

    # Limitar a color naranja
    orange_mask = tf.logical_and(h > 0.03, h < 0.12)
    orange_mask = tf.logical_and(orange_mask, s > 0.4)
    orange_mask = tf.logical_and(orange_mask, v > 0.3)
    orange_mask = tf.cast(orange_mask, tf.float32)
    orange_mask = tf.repeat(orange_mask, 3, axis=-1)

    image = image
    image = image * 0.8 + orange_mask * 0.2

    return image, label


for categoria in parametros:
    print(f"iniciando entrenamiento de {categoria}")

    num_labels = len(parametros[categoria]['labels'])

    (train_dataset, validation_dataset) = tf.keras.preprocessing.image_dataset_from_directory(
        f'./dataset/{categoria}',
        label_mode='int',
        validation_split=0.3,
        subset='both',
        seed=42,
        image_size=(228, 228),
    )

    if categoria == "health":
        train_dataset = train_dataset.map(preprocess_health)
        validation_dataset = validation_dataset.map(preprocess_health)

    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(228, 228, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    if categoria == "health":
        model = tf.keras.models.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_labels, activation='softmax')
        ])
    else:
        model = tf.keras.models.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(num_labels, activation='softmax')
        ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_dataset,
        epochs=5 if categoria != "health" else 30,
        validation_data=validation_dataset,
    )
    model.save(f'models/{categoria}.keras')
