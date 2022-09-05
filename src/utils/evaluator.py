import os
import yaml
import tensorflow as tf

class Evaluator:
    def __init__(self, test, test_labels):
        self.test = test
        self.test_labels = test_labels
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.config_path = os.path.join(self.dir_path, 'directories.yaml')

    def evaluate(self):
        with open(self.config_path, 'r') as file:
            directories = yaml.safe_load(file)

        export_dir = directories['export_dir']
        export_dir = os.path.join(self.dir_path, export_dir)

        reloaded = tf.keras.models.load_model(export_dir)

        loss, acc = reloaded.evaluate(
                    self.test, self.test_labels,
                    batch_size=32)

        return loss, acc

