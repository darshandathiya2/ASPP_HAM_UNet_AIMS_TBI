
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

def train_model(model,
                train_dataset,
                val_dataset,
                epochs=100):

    callbacks = [
        EarlyStopping(
            patience=10,
            restore_best_weights=True
        ),

        ReduceLROnPlateau(
            factor=0.1,
            patience=5
        ),

        ModelCheckpoint(
            "best_model.h5",
            save_best_only=True
        )
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks
    )

    return history
