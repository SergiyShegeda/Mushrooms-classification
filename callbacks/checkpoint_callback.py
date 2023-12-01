from keras.callbacks import ModelCheckpoint
from datetime import datetime
import os
import shutil


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, dataset: str = 'none', *args, **kwargs):
        self.dataset = dataset
        super().__init__(*args, **kwargs, filepath='')

    def on_epoch_end(self, epoch, logs=None):
        if epoch and epoch % self.save_freq == 0:
            filepath = os.path.join(
                self._get_checkpoint_dir(),
                f"weights.{epoch:02d}-{logs['val_loss']:.2f}.keras" if 'val_loss' in logs else f"weights.{epoch:02d}.keras"
            )
            self.filepath = filepath
            super().on_epoch_end(epoch, logs)

    def _get_checkpoint_dir(self):
        current_date = datetime.now().strftime("%Y-%m-%d")
        checkpoint_dir = "checkpoints/{}/{}/".format(self.dataset, current_date)
        shutil.rmtree('checkpoints/', ignore_errors=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        return checkpoint_dir
