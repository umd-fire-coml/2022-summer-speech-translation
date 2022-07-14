from translation import *
# steven's addition: saving checkpoints
checkpoint_path = "./EngToSpanishckpts/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

epochs = 20  # This should be at least 30 for convergence

transformer.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[cp_callback])