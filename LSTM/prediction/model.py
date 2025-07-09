import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, output_dim,
                 dropout_rate=0.2, bidirectional=True):
        """
        初始化 LSTM 模型

        :param vocab_size: 词汇表大小，这里为 4（A, C, G, U）
        :param embedding_dim: 嵌入层的维度
        :param hidden_dim: LSTM 隐藏层的维度
        :param num_layers: LSTM 层数
        :param output_dim: 输出层的维度，这里预测半衰期，为 1
        :param dropout_rate: Dropout 概率，用于防止过拟合
        :param bidirectional: 是否使用双向 LSTM
        """
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout_rate, bidirectional=bidirectional)
        # 如果是双向 LSTM，隐藏层输出维度需要乘以 2
        self.fc1 = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        前向传播过程

        :param x: 输入的 mRNA 序列编码
        :return: 预测的半衰期
        """
        # 嵌入层
        embedded = self.embedding(x)
        # LSTM 层
        lstm_out, _ = self.lstm(embedded)
        # 取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]
        # 第一个全连接层，使用 ReLU 激活函数
        fc1_out = self.fc1(lstm_out)
        fc1_out = F.relu(fc1_out)
        # Dropout 层，防止过拟合
        fc1_out = self.dropout(fc1_out)
        # 第二个全连接层，输出预测结果
        output = self.fc2(fc1_out)
        return output
