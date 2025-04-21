import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from matplotlib import pyplot as plt
from Neural_network import CNN, RNN
from prework import get_batch

# 选择设备：优先 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# 训练、评估函数

def NN_embedding(model, train_loader, test_loader, learning_rate, epoch):
    # 模型搬到 device
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = F.cross_entropy

    train_loss, test_loss, long_loss = [], [], []
    train_acc, test_acc, long_acc = [], [], []

    for it in range(epoch):
        model.train()
        for x_batch, y_batch in train_loader:
            # 如果 x_batch/y_batch 不是 Tensor，可统一转换：
            if not isinstance(x_batch, torch.Tensor):
                x_batch = torch.tensor(x_batch, dtype=torch.long, device=device)
            else:
                x_batch = x_batch.to(device)
            if not isinstance(y_batch, torch.Tensor):
                y_batch = torch.tensor(y_batch, dtype=torch.long, device=device)
            else:
                y_batch = y_batch.to(device)

            pred = model(x_batch)
            optimizer.zero_grad()
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()

        # 评估
        model.eval()
        tmp_tr_loss, tmp_te_loss, tmp_long_loss = 0.0, 0.0, 0.0
        tmp_tr_acc, tmp_te_acc, tmp_long_acc = [], [], []
        length_threshold = 20

        with torch.no_grad():
            # 训练集评估
            for x_batch, y_batch in train_loader:
                if not isinstance(x_batch, torch.Tensor):
                    x_batch = torch.tensor(x_batch, dtype=torch.long, device=device)
                else:
                    x_batch = x_batch.to(device)
                if not isinstance(y_batch, torch.Tensor):
                    y_batch = torch.tensor(y_batch, dtype=torch.long, device=device)
                else:
                    y_batch = y_batch.to(device)

                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                tmp_tr_loss += loss.item()
                y_pred = pred.argmax(dim=1)
                acc = (y_pred == y_batch).float().mean().item()
                tmp_tr_acc.append(acc)

            # 测试集评估
            for x_batch, y_batch in test_loader:
                if not isinstance(x_batch, torch.Tensor):
                    x_batch = torch.tensor(x_batch, dtype=torch.long, device=device)
                else:
                    x_batch = x_batch.to(device)
                if not isinstance(y_batch, torch.Tensor):
                    y_batch = torch.tensor(y_batch, dtype=torch.long, device=device)
                else:
                    y_batch = y_batch.to(device)

                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                tmp_te_loss += loss.item()
                y_pred = pred.argmax(dim=1)
                acc = (y_pred == y_batch).float().mean().item()
                tmp_te_acc.append(acc)

                # 长句子评估
                if x_batch.size(1) > length_threshold:
                    tmp_long_loss += loss.item()
                    tmp_long_acc.append(acc)

        # 计算平均
        avg_tr_loss = tmp_tr_loss / len(tmp_tr_acc)
        avg_te_loss = tmp_te_loss / len(tmp_te_acc)
        avg_long_loss = tmp_long_loss / len(tmp_long_acc) if tmp_long_acc else 0.0
        avg_tr_acc = sum(tmp_tr_acc) / len(tmp_tr_acc)
        avg_te_acc = sum(tmp_te_acc) / len(tmp_te_acc)
        avg_long_acc = sum(tmp_long_acc) / len(tmp_long_acc) if tmp_long_acc else 0.0

        train_loss.append(avg_tr_loss)
        test_loss.append(avg_te_loss)
        long_loss.append(avg_long_loss)
        train_acc.append(avg_tr_acc)
        test_acc.append(avg_te_acc)
        long_acc.append(avg_long_acc)

        print(f"Epoch {it+1}/{epoch}")
        print(f" Train loss: {avg_tr_loss:.4f}, acc: {avg_tr_acc:.4f}")
        print(f" Test  loss: {avg_te_loss:.4f}, acc: {avg_te_acc:.4f}")
        print(f" Long  loss: {avg_long_loss:.4f}, acc: {avg_long_acc:.4f}\n")

    return train_loss, test_loss, long_loss, train_acc, test_acc, long_acc


# 绘图主函数

def plot(random_embedding, glove_embedding, learning_rate, batch_size, epoch):
    # DataLoader
    train_random = get_batch(random_embedding.train_matrix,
                             random_embedding.train_y,
                             batch_size)
    test_random = get_batch(random_embedding.test_matrix,
                            random_embedding.test_y,
                            batch_size)
    train_glove = get_batch(glove_embedding.train_matrix,
                            glove_embedding.train_y,
                            batch_size)
    test_glove = get_batch(glove_embedding.test_matrix,
                           glove_embedding.test_y,
                           batch_size)

    # 实例化并搬到 device
    torch.manual_seed(2025)
    random_rnn = RNN(50, 50, random_embedding.len).to(device)
    torch.manual_seed(2025)
    random_cnn = CNN(50, random_embedding.len, random_embedding.longest).to(device)
    torch.manual_seed(2025)
    glove_rnn = RNN(50, 50, glove_embedding.len,
                    weight=torch.tensor(glove_embedding.embedding,
                                        dtype=torch.float,
                                        device=device)).to(device)
    torch.manual_seed(2025)
    glove_cnn = CNN(50, glove_embedding.len,
                    glove_embedding.longest,
                    weight=torch.tensor(glove_embedding.embedding,
                                        dtype=torch.float,
                                        device=device)).to(device)

    # 训练 & 评估
    trl_ran_rnn, tel_ran_rnn, lol_ran_rnn, tra_ran_rnn, tes_ran_rnn, lon_ran_rnn = \
        NN_embedding(random_rnn, train_random, test_random, learning_rate, epoch)
    trl_ran_cnn, tel_ran_cnn, lol_ran_cnn, tra_ran_cnn, tes_ran_cnn, lon_ran_cnn = \
        NN_embedding(random_cnn, train_random, test_random, learning_rate, epoch)
    trl_glo_rnn, tel_glo_rnn, lol_glo_rnn, tra_glo_rnn, tes_glo_rnn, lon_glo_rnn = \
        NN_embedding(glove_rnn, train_glove, test_glove, learning_rate, epoch)
    trl_glo_cnn, tel_glo_cnn, lol_glo_cnn, tra_glo_cnn, tes_glo_cnn, lon_glo_cnn = \
        NN_embedding(glove_cnn, train_glove, test_glove, learning_rate, epoch)

    # 绘制 Loss 和 Accuracy 曲线
    epochs = list(range(1, epoch + 1))
    plt.figure(figsize=(8, 8))

    # Train/Test Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, trl_ran_rnn, 'r--', label='RNN+random')
    plt.plot(epochs, trl_ran_cnn, 'g--', label='CNN+random')
    plt.plot(epochs, trl_glo_rnn, 'b--', label='RNN+glove')
    plt.plot(epochs, trl_glo_cnn, 'y--', label='CNN+glove')
    plt.title("Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(fontsize=8)

    plt.subplot(2, 2, 2)
    plt.plot(epochs, tel_ran_rnn, 'r--', label='RNN+random')
    plt.plot(epochs, tel_ran_cnn, 'g--', label='CNN+random')
    plt.plot(epochs, tel_glo_rnn, 'b--', label='RNN+glove')
    plt.plot(epochs, tel_glo_cnn, 'y--', label='CNN+glove')
    plt.title("Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(fontsize=8)

    # Train/Test Accuracy
    plt.subplot(2, 2, 3)
    plt.plot(epochs, tra_ran_rnn, 'r--', label='RNN+random')
    plt.plot(epochs, tra_ran_cnn, 'g--', label='CNN+random')
    plt.plot(epochs, tra_glo_rnn, 'b--', label='RNN+glove')
    plt.plot(epochs, tra_glo_cnn, 'y--', label='CNN+glove')
    plt.title("Train Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend(fontsize=8)

    plt.subplot(2, 2, 4)
    plt.plot(epochs, tes_ran_rnn, 'r--', label='RNN+random')
    plt.plot(epochs, tes_ran_cnn, 'g--', label='CNN+random')
    plt.plot(epochs, tes_glo_rnn, 'b--', label='RNN+glove')
    plt.plot(epochs, tes_glo_cnn, 'y--', label='CNN+glove')
    plt.title("Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('main_plot.jpg')
    plt.show()

    # Long sentence curves
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, lon_ran_rnn, 'r--', label='RNN+random')
    plt.plot(epochs, lon_ran_cnn, 'g--', label='CNN+random')
    plt.plot(epochs, lon_glo_rnn, 'b--', label='RNN+glove')
    plt.plot(epochs, lon_glo_cnn, 'y--', label='CNN+glove')
    plt.title("Long Sentence Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend(fontsize=8)

    plt.subplot(2, 1, 2)
    plt.plot(epochs, lol_ran_rnn, 'r--', label='RNN+random')
    plt.plot(epochs, lol_ran_cnn, 'g--', label='CNN+random')
    plt.plot(epochs, lol_glo_rnn, 'b--', label='RNN+glove')
    plt.plot(epochs, lol_glo_cnn, 'y--', label='CNN+glove')
    plt.title("Long Sentence Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig('sub_plot.jpg')
    plt.show()