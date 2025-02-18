from utils import dataset_helpers


dataset_helpers.create_new_dataset("/content/data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData", "/content/newData")

dataset_helpers.startBinaryTraining("../newData")