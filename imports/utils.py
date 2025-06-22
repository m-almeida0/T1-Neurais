import matplotlib.pyplot as plt
import torch

def plot_trainval_graphs(train_losses, train_accs, val_losses, val_accs):
  epochs_range = range(1, len(train_losses) + 1)

  plt.figure(figsize=(8, 4))
  plt.plot(epochs_range, train_losses, label='Train Loss')
  plt.plot(epochs_range, val_losses, label='Val Loss')
  plt.xlabel('Época')
  plt.ylabel('Loss')
  plt.title('Loss por Época')
  plt.legend()
  plt.grid(True)
  plt.show()

  plt.figure(figsize=(8, 4))
  plt.plot(epochs_range, train_accs, label='Train Acc')
  plt.plot(epochs_range, val_accs, label='Val Acc')
  plt.xlabel('Época')
  plt.ylabel('Acurácia')
  plt.title('Acurácia por Época')
  plt.legend()
  plt.grid(True)
  plt.show()

def train(model,optimizer,criterion,train_loader,val_loader,num_epochs=10, device = torch.device("cuda")):
    """
    Executa o treinamento e validação por num_epochs e depois
    retorna 4 listas: train_losses, train_accs, val_losses e val_accs.
    """
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        # Itera sobre batches de treino, sendo que o loader retorna (images, labels, caption)
        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Acumula loss
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size

            # Acumula acertos para train acc
            _, preds = torch.max(outputs, dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += batch_size


        epoch_train_loss = running_loss / running_total
        epoch_train_acc = running_correct / running_total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")

        ## Etapa de avaliação
        model.eval()
        val_running_loss = 0.0
        val_running_correct = 0
        val_running_total = 0

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, labels)

                batch_size = images.size(0)
                val_running_loss += loss.item() * batch_size

                _, preds = torch.max(outputs, dim=1)
                val_running_correct += (preds == labels).sum().item()
                val_running_total += batch_size

        epoch_val_loss = val_running_loss / val_running_total
        epoch_val_acc = val_running_correct / val_running_total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        print(f"           Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
    return train_losses, train_accs, val_losses, val_accs