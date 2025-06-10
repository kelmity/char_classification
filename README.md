Cartoon Character Classification

This project focuses on image classification using deep learning, specifically a convolutional neural network (CNN), to classify cartoon characters from images. The dataset consists of labeled images of cartoon characters, which are used to train the model. The project utilizes PyTorch and torchvision libraries to implement the model and train it on the dataset.

Requirements

Before running the project, make sure to install the following dependencies:

Python

PyTorch

torchvision

numpy

matplotlib

tqdm

google.colab (for Google Colab users)


You can install the necessary libraries by running:

pip install torch torchvision numpy matplotlib tqdm

Project Setup

1. Clone this repository or upload the notebook to your Google Colab workspace.


2. Ensure that the dataset is stored in the correct directory. You can either use Google Drive or store the dataset locally.


3. Mount Google Drive (if using Colab):

from google.colab import drive
drive.mount('/content/drive')



Dataset

The dataset used for training is the Cartoon Character Dataset, where each image represents a cartoon character. The images are categorized into classes based on the character they represent.

Training data location: /content/drive/MyDrive/archive/cartoon/train/

The dataset is loaded using the ImageFolder utility from PyTorch, which automatically assigns labels to the images based on the folder structure.


Hyperparameters

Image Size: 64x64

Batch Size: 64

Normalization Stats: (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


These hyperparameters are used to preprocess the images for training.

Model Training

The model uses a basic convolutional neural network (CNN) architecture for classifying images. The following steps are involved in the training process:

1. Load the Data: The training data is loaded using the DataLoader class in PyTorch.


2. Preprocessing: Images are resized to 64x64 pixels, cropped, and normalized.


3. Training: The model is trained on the dataset using a batch size of 64, with appropriate optimizer and loss function.



trainloader = datasets.ImageFolder(dataroot, transform=T.Compose([
    T.Resize(image_size),
    T.CenterCrop(image_size),
    T.ToTensor(),
    T.Normalize(*stats)
]))

Running the Model

To start training the model, simply run the cells provided in the notebook. If you're using Google Colab, make sure to mount your Google Drive and update the path to the dataset accordingly.