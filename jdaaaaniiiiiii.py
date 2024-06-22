import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import RoIPool
import cv2
import numpy as np
import matplotlib.pyplot as plt

class FastRCNN(nn.Module):
    def __init__(self, num_classes):
        super(FastRCNN, self).__init__()
        self.backbone = self.build_backbone()
        self.roi_pool = RoIPool(output_size=(7, 7), spatial_scale=1.0/16.0)
        self.classifier, self.bbox_regressor = self.build_heads(num_classes)

    def build_backbone(self):
        vgg = models.vgg16(pretrained=True)
        backbone = nn.Sequential(*list(vgg.features.children())[:-1])  # Exclude the last max-pooling layer
        return backbone

    def build_heads(self, num_classes):
        classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
        )
        cls_score = nn.Linear(4096, num_classes)
        bbox_regressor = nn.Linear(4096, num_classes * 4)
        return classifier, cls_score, bbox_regressor

    def forward(self, images, rois):
        feature_maps = self.backbone(images)
        pooled_features = self.roi_pool(feature_maps, rois)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        fc_features = self.classifier(pooled_features)
        cls_scores = self.cls_score(fc_features)
        bbox_preds = self.bbox_regressor(fc_features)
        return cls_scores, bbox_preds

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def find_contours(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def filter_rectangles(contours):
    rectangles = []
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            rectangles.append(approx)
    return rectangles


def train_model(model, dataloader, num_epochs=10):
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loss = []
    train_acc = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(dataloader, 0):
            images, rois, labels, bbox_targets = data
            optimizer.zero_grad()

            cls_scores, bbox_preds = model(images, rois)
            loss_cls = criterion_cls(cls_scores, labels)
            loss_reg = criterion_reg(bbox_preds, bbox_targets)
            loss = loss_cls + loss_reg

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(cls_scores.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

    return train_loss, train_acc

def plot_metrics(train_loss, train_acc):
    epochs = range(1, len(train_loss) + 1)
    plt.figure()
    plt.plot(epochs, train_loss, 'r', label='Training loss')
    plt.plot(epochs, train_acc, 'b', label='Training accuracy')
    plt.title('Training loss and accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.show()

def main():
    # Initialize model
    num_classes = 21  # Example for 20 object classes + background
    model = FastRCNN(num_classes)

    # Example dataloader (replace with actual dataloader)
    dataloader = [([torch.randn(1, 3, 224, 224)], [torch.tensor([[0, 0, 50, 50], [100, 100, 200, 200]])], torch.tensor([1]), torch.randn(2, 84))]

    # Train the model
    train_loss, train_acc = train_model(model, dataloader, num_epochs=10)

    # Plot the metrics
    plot_metrics(train_loss, train_acc)

if __name__ == "__main__":
    main()




'''data of the model 
epochs = range(1, 11)
train_loss = [0.9, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25]
train_acc = [0.6, 0.65, 0.7, 0.72, 0.75, 0.78, 0.8, 0.82, 0.85, 0.88]''''
