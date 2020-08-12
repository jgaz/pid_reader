import unittest
from pathlib import Path

from generator.config import DIAGRAM_PATH
from generator.metadata import TrainingDatasetLabelDictionaryStorage
from generator.metadata import DiagramSymbolsStorage


class TestGenerateSetDiagrams(unittest.TestCase):
    def test_raw_class_indexing(self):
        """
        Check classes tagging in raw diagrams
        :return:
        """
        # Collect raw diagrams classes
        class_dictionary = TrainingDatasetLabelDictionaryStorage.get(DIAGRAM_PATH)

        # Collect classes in mapping file
        classes = {}
        diagram_path = Path(DIAGRAM_PATH)
        dss = DiagramSymbolsStorage()
        for filename in diagram_path.glob("*.pickle"):
            # Get the pickle information
            metadata = dss.load(filename=filename)
            classes[metadata[0].name] = 0

        # Make sure they match in both directions
        self.assertTrue(
            set(classes.keys()).difference(set(class_dictionary.keys())) == set()
        )

    def test_tensorflow_class_indexing(self):
        """
        import tensorflow as tf
        raw_dataset = tf.data.TFRecordDataset("00001-of-00020.tfrecord")
        record = raw_dataset.take(1)
        example = tf.train.Example()
        example.ParseFromString(list(record.as_numpy_iterator())[0])
        """


if __name__ == "__main__":
    unittest.main()
