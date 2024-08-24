
import keras

class CustomModelCheckpoint(keras.callbacks.Callback):

    def __init__(self, models, path, saver_freq):
        self.models = models
        self.path = path
        self.save_freq = saver_freq

    def on_epoch_end(self, epoch, logs=None):
        # logs is a dictionary
        if(epoch % self.save_freq == 0):
            print(f"Saving epoch: {epoch}, train_acc: {logs['acc']}, : {logs['acc_seq']}")
            for name in self.models:
                model = self.models[name]
                model.save(f"{self.path}/{name}.keras", overwrite=True)


def acc_seq(y_true, y_pred):
    return keras.ops.mean(keras.ops.min(keras.ops.equal(keras.ops.argmax(y_true, axis=-1),
                  keras.ops.argmax(y_pred, axis=-1)), axis=-1))

