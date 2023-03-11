import os
import time
from PIL import Image, ImageOps
from torch import nn
import torchvision.transforms as T
import torch
import cv2
import numpy as np
import streamlit as st

st.set_page_config(layout="wide", page_title="Digit Recognition")

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0))

        self.fully_connected1 = nn.Linear(in_features=120, out_features=84)
        self.fully_connected2 = nn.Linear(in_features=84, out_features=10)

        self.pooling_layer = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Convolution Layer 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pooling_layer(x)

        # Convolution Layer 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pooling_layer(x)
        x = self.dropout(x)

        # Convolution Layer 3
        x = self.conv3(x)
        x = self.relu(x)

        # flatten x
        x = x.view(-1, 120)

        # Fully connected layer 1
        x = self.fully_connected1(x)
        x = self.relu(x)

        # Fully connected layer 2
        x = self.fully_connected2(x)

        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Network()
model.load_state_dict(torch.load('mnist_model.pth', map_location=torch.device(device)))

st.title("MNIST Image Classification")
st.subheader("This is a simple image classification web application to predict handwritten digits")

st.sidebar.write('## Please upload an image file :camera:', unsafe_allow_html=True)
file = st.sidebar.file_uploader("## Upload", type=["png"])

if file is None:
    imagefile = './0.png'
else:
    imagefile = file

img = Image.open(imagefile)
img_copy = img
img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
transform = T.Compose([
    T.ToTensor(),
    T.Resize((28, 28))
])
img = transform(img)
st.image(img_copy, width=150)
model.eval()
results = model(img)
category = torch.argmax(results)
print(category.numpy())
st.write('<hr font-size: 30px;>The image is digit </hr>', str(category.numpy()), unsafe_allow_html=True)
