
def get_acc_loss(log_path):

    train_acc, train_loss, val_acc, val_loss = [], [], [], []

    with open(log_path, 'r') as f:
        epochs_history = [line.strip() for line in f.readlines()]
        epochs_history = [item.split(',') for item in epochs_history]
        for epoch_history in epochs_history:
            for history in epoch_history:
                items = history.strip().split(':')
                if items[0] == 'loss':
                    train_loss.append(float(items[1]))
                elif items[0] == 'acc':
                    train_acc.append(float(items[1]))
                elif items[0] == 'val_loss':
                    val_loss.append(float(items[1]))
                elif items[0] == 'val_acc':
                    val_acc.append(float(items[1]))

    return train_acc, train_loss, val_acc, val_loss


