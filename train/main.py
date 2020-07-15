from train.model import Net
from train.puzzle_dataset import PuzzleDataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
import torch.optim as optim
import random

if __name__ == "__main__":
    # %%

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net()
    model.to(device)

    # %%

    summary(model, (3, 384, 384))

    # %%

    epoch_num = 1260
    batch_size = 64
    train_data = PuzzleDataset("../train-data")
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    loss_func = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
    # %%

    for epoch in range(epoch_num):
        train_loss = 0
        model.train()
        for batch_image, batch_label in train_loader:
            inputs, labels = batch_image.to(device), batch_label.to(device)
            optimizer.zero_grad()  # Clear optimizers
            output = model.forward(inputs)
            # print(inputs.shape, labels.shape, output.shape)
            # Forward pass
            loss = loss_func(output, labels)  # Loss
            loss.backward()  # Calculate gradients (backpropogation)
            optimizer.step()  # Adjust parameters based on gradients
            train_loss += loss.item() * inputs.size(0)  # Add the loss to the training set's rnning loss

        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

        inputs, labels = train_data.__getitem__(random.randrange(25, 360))
        inputs, _ = inputs.to(device), labels.to(device)  # Move to device
        inputs = inputs[np.newaxis, :]
        print(inputs.shape, labels.shape)
        output = model.forward(inputs)  # Forward pass
        print(labels)
        print(output)

        if epoch % 100 == 0:
            torch.onnx.export(model,  # model being run
                              inputs,  # model input (or a tuple for multiple inputs)
                              "model-{0}.onnx".format(epoch),
                              # where to save the model (can be a file or file-like object)
                              export_params=True,  # store the trained parameter weights inside the model file
                              opset_version=10,  # the ONNX version to export the model to
                              do_constant_folding=True,  # whether to execute constant folding for optimization
                              input_names=['input'],  # the model's input names
                              output_names=['output'],  # the model's output names
                              dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                            'output': {0: 'batch_size'}})
