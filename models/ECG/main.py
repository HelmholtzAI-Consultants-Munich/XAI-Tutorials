import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from ResNet1D import ResNet, ResNetBlock
from ECG import ECG



def training_loop(model, train_loader, criterion, optimizer, num_epochs, writer, device):
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs[0], labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
        
        torch.save(model.state_dict(), 'weights/model_weights'+str(epoch)+'.pth')

        writer.add_scalar('training loss', running_loss/len(train_loader), epoch)

    torch.save(model.state_dict(), 'weights/model_final_weights_ecg.pth')

def inference_loop(model, test_loader, criterion):

    
    model.eval()

    device = next(model.parameters()).device
    true_positives = [0] * 5
    false_positives = [0] * 5
    false_negatives = [0] * 5
    true_negatives = [0] * 5
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            targets = targets.to(device)
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs[0]

            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            total_samples += inputs.size(0)
            _, predictions = torch.max(outputs, -1)

            for i in range(5):

                true_positives[i] += ((predictions == i) & (targets == i)).sum().item()
                false_positives[i] += ((predictions == i) & (targets != i)).sum().item()
                false_negatives[i] += ((predictions != i) & (targets == i)).sum().item()
                true_negatives[i] += ((predictions != i) & (targets != i)).sum().item()
    
    accuracy = (sum(true_positives) + sum(true_negatives)) / (sum(true_positives) + sum(false_positives) + sum(false_negatives) + sum(true_negatives))
    recall = [true_positives[i] / (true_positives[i] + false_negatives[i]) if true_positives[i] + false_negatives[i] > 0 else 0 for i in range(5)]
    precision = [true_positives[i] / (true_positives[i] + false_positives[i]) if true_positives[i] + false_positives[i] > 0 else 0 for i in range(5)]
    f1 = [2 * precision[i] * recall[i] / (precision[i] + recall[i]) if precision[i] + recall[i] > 0 else 0 for i in range(5)]
    average_loss = total_loss / total_samples
    
    return accuracy, recall, precision, f1, average_loss


if __name__ == "__main__":

    model_weights_path = 'weights/model_final_weights_ecg.pth'
    train_data_path = 'data/Dataset_ECG/mitbih_train.csv'
    test_data_path = 'data/Dataset_ECG/mitbih_test.csv'
    num_epochs = 10

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ResNet(ResNetBlock, [2, 2, 2, 2], num_classes=5)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    ecg = ECG(train_data_path)
    train_loader = DataLoader(ecg, batch_size=64, shuffle=True, num_workers=4)

    ecg = ECG(test_data_path)
    test_loader = DataLoader(ecg, batch_size=64, shuffle=True, num_workers=4)

    writer = SummaryWriter()

    # training_loop(model, train_loader, criterion, optimizer, num_epochs, writer, device)

    model.load_state_dict(torch.load(model_weights_path))

    accuracy, recall, precision, f1, average_loss = inference_loop(model, test_loader, criterion)

    print("Accuracy: ", accuracy)
    print("Recall: ", recall)
    print("Precision: ", precision)
    print("F1: ", f1)
    print("Average loss: ", average_loss)

