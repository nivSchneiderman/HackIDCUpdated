from PIL import Image
from enum import Enum
import base64
import io
import server.learningAlgorithm
import numpy as np


class Classification(Enum):
    A = 1,
    B = 2,
    C = 3


class AppImage:
    def __init__(self, request_image_file_data, model, graph, counter):
        _, image_string = request_image_file_data.json.split(',')
        self.image_bytes = base64.b64decode(str.encode(image_string))
        self.image_colored = Image.open(io.BytesIO(self.image_bytes))
        # self.image_colored.save('C:/repo/Dynamic_Data/Class_C/Class_C_{0}.jpg'.format(str(counter)))
        self.image_grayscale = self.image_colored.convert('LA')
        self._classification = None
        self._model = model
        self._graph = graph

    def get_classification(self):
        test_images = [np.asarray(self.image_colored)]
        test_images = np.array(test_images).astype('float32') / 255
        with self._graph.as_default():
            string_class = server.learningAlgorithm.get_labels_from_class_numbers\
                (self._model.predict_classes([test_images]))[0]
            return {
                'A': Classification.A,
                'B': Classification.B,
                'C': Classification.C
            }[string_class]

    def classify_image(self):
        if self._classification is None:
            self._classification = self.get_classification()
        return self._classification

    def show_colored(self):
        self.image_colored.show()

    def show_grayscale(self):
        self.image_grayscale.show()

    @staticmethod
    def explain_classification(classification):
        return {
            Classification.A: 'Good posture',
            Classification.B: 'Too far',
            Classification.C: 'Too close'
        }[classification]

