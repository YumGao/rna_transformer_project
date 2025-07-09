import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_dataloader
from model import LSTMModel

# 示例数据
sequences = [
    ['A', 'U', 'A', 'C', 'G', 'C', 'A', 'A', 'G', 'G', 'A', 'C', 'C', 'G', 'A', 'U', 'C', 'G', 'G', 'U'],
    ['C', 'U', 'C', 'U', 'C', 'C', 'A', 'G', 'A', 'G', 'C', 'C', 'G', 'U', 'U', 'U', 'G', 'C', 'A', 'G'],
    ['A', 'C', 'U', 'G', 'G', 'G', 'C', 'U', 'C', 'A', 'G', 'G', 'U', 'U', 'U', 'G', 'U', 'U', 'C', 'C']
]
# 假设的半衰期数据
half_lives = [1.5, 2.0, 2.5]

# 数据加载
dataloader = get_dataloader(sequences, half_lives, batch_size=1)

# 模型参数
vocab_size = 4
embedding_dim = 32
hidden_dim = 64
num_layers = 2
output_dim = 1
dropout_rate = 0.2
bidirectional = True

# 初始化模型
model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers, output_dim,
                  dropout_rate=dropout_rate, bidirectional=bidirectional)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for sequences, half_lives in dataloader:
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs.squeeze(), half_lives)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}')

# 保存模型
torch.save(model.state_dict(), 'lstm_model.pth')


