import torch
import torch.nn as nn
from torchviz import make_dot
from torchsummary import summary

# Define the EncoderDecoderCNN model
class EncoderDecoderCNN(nn.Module):
    def __init__(self):
        super(EncoderDecoderCNN, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu4 = nn.ReLU()
        self.t_conv2 = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu5 = nn.ReLU()
        self.final_conv = nn.Conv2d(32, 6, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(dim=1)
        self.t_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.relu1(self.conv1(x))
        x2 = self.pool1(x1)
        x3 = self.relu2(self.conv2(x2))
        x4 = self.pool2(x3)
        x5 = self.relu3(self.conv3(x4))

        x6 = self.relu4(self.t_conv1(x5))
        x7 = torch.cat((x3, x6), 1)

        x8 = self.relu4(self.t_conv2(x7))
        x9 = torch.cat((x8, x1), 1)

        x10 = self.relu4(self.t_conv3(x9))
        output = self.softmax(self.final_conv(x10))
        return output


# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model and move it to the correct device
model = EncoderDecoderCNN().to(device)

# Define a dummy input tensor (Batch size: 1, Channels: 5, Height: 224, Width: 224)
dummy_input = torch.randn(1, 5, 224, 224).to(device)

# Create a graphical representation of the model using torchviz
output = model(dummy_input)  # Perform a forward pass to generate the graph
dot_graph = make_dot(output, params=dict(model.named_parameters()))

# Save the graph as a PNG file (optional)
dot_graph.render("encoder_decoder_model", format="png")

# Display the graph in a Jupyter Notebook or similar environment
dot_graph.view()

# Print a textual summary of the model using torchsummary
summary(model, input_size=(5, 224, 224))