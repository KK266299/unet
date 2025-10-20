import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
from utils import extract_time, MinMaxScaler,  train_test_divide, batch_generator, pad_sequences

def discriminative_score_metrics (ori_data, generated_data):
    no, seq_len, dim = np.asarray(ori_data).shape

    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])  

    hidden_dim = int(dim/2)
    iterations = 2000
    batch_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========== 输入占位说明（对应TensorFlow的placeholder） ==========
    # X:     [None, max_seq_len, dim]  - 真实数据输入 (myinput_x)
    # X_hat: [None, max_seq_len, dim]  - 生成数据输入 (myinput_x_hat)
    # T:     [None]                    - 真实数据序列长度 (myinput_t)
    # T_hat: [None]                    - 生成数据序列长度 (myinput_t_hat)
    # 
    # 注意：PyTorch不需要预定义placeholder，在训练循环中直接使用tensor
    # ================================================================

    class Discriminator(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super(Discriminator, self).__init__()
            self.gru = nn.GRU(
                input_size=input_dim, 
                hidden_size=hidden_dim, 
                num_layers=1, 
                batch_first=True
                )

            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x, t):
            packed_input = nn.utils.rnn.pack_padded_sequence(
                x,
                t.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
            d_output, hidden_pre = self.gru(packed_input)
            d_last_states = hidden_pre.squeeze(0)

            y_hat_logit = self.fc(d_last_states)
            y_hat = torch.sigmoid(y_hat_logit)
            d_vars = list(self.parameters())

            return y_hat_logit, y_hat, d_vars

    discriminator = Discriminator(dim, hidden_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(discriminator.parameters())
    train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
        train_test_divide(ori_data, generated_data, ori_time, generated_time)
    
    discriminator.train()
    for it in range(iterations):
        X_mb, T_mb = batch_generator(train_x, train_t, batch_size)
        X_hat_mb, T_hat_mb = batch_generator(train_x_hat, train_t_hat, batch_size)

        """
        #padding
        packed_input = nn.utils.rnn.pack_padded_sequence(
                x,
                t.cpu(),
                batch_first=True,
                enforce_sorted=False
            ) 因为pytorch无法像tf直接处理变长序列,先填充在用这个函数剔除
        # Padding 
        """

        X_mb_padded = pad_sequences(X_mb, max_seq_len)
        X_hat_mb_padded = pad_sequences(X_hat_mb, max_seq_len)

        X = torch.FloatTensor(X_mb_padded).to(device)         # myinput_x
        X_hat = torch.FloatTensor(X_hat_mb_padded).to(device) # myinput_x_hat
        T = torch.LongTensor(T_mb).to(device)                 # myinput_t
        T_hat = torch.LongTensor(T_hat_mb).to(device)         # myinput_t_hat

        y_logit_real, _, _ = discriminator(X, T)
        y_logit_fake, _, _ = discriminator(X_hat, T_hat)

        labels_real = torch.ones_like(y_logit_real)
        labels_fake = torch.zeros_like(y_logit_fake)
        d_loss_real = criterion(y_logit_real, labels_real)
        d_loss_fake = criterion(y_logit_fake, labels_fake)
        d_loss = d_loss_real + d_loss_fake

        optimizer.zero_grad()
        d_loss.backward()
        optimizer.step()
    
    discriminator.eval()
    with torch.no_grad():
        test_x_padded = pad_sequences(test_x, max_seq_len)
        test_x_hat_padded = pad_sequences(test_x_hat, max_seq_len)
        
        X = torch.FloatTensor(test_x_padded).to(device)
        X_hat = torch.FloatTensor(test_x_hat_padded).to(device)
        T = torch.LongTensor(test_t).to(device)
        T_hat = torch.LongTensor(test_t_hat).to(device)
        _, y_pred_real, _ = discriminator(X, T)
        _, y_pred_fake, _ = discriminator(X_hat, T_hat)

        y_pred_real_curr = y_pred_real.cpu().numpy()
        y_pred_fake_curr = y_pred_fake.cpu().numpy()

        y_pred_final = np.squeeze(np.concatenate((y_pred_real_curr, y_pred_fake_curr), axis=0))
        y_label_final = np.concatenate((np.ones([len(y_pred_real_curr),]), 
                                     np.zeros([len(y_pred_fake_curr),])), axis=0)
        acc = accuracy_score(y_label_final, (y_pred_final > 0.5))
        discriminative_score = np.abs(0.5 - acc)
    
    return discriminative_score
