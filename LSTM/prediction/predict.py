import torch
from data_loader import BASE_MAPPING
from model import LSTMModel

# 加载模型
vocab_size = 4
embedding_dim = 32
hidden_dim = 64
num_layers = 2
output_dim = 1
dropout_rate = 0.2
bidirectional = True

model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers, output_dim,
                  dropout_rate=dropout_rate, bidirectional=bidirectional)
model.load_state_dict(torch.load('lstm_model.pth'))
model.eval()

# 示例预测
sequence = ['A', 'U', 'A', 'C', 'G', 'C', 'A', 'A', 'G', 'G', 'A', 'C', 'C', 'G', 'A', 'U', 'C', 'G', 'G', 'U']
encoded_sequence = [BASE_MAPPING[base] for base in sequence]
encoded_sequence = torch.tensor(encoded_sequence, dtype=torch.long).unsqueeze(0)

with torch.no_grad():
    output = model(encoded_sequence)
    predicted_half_life = output.item()

print(f'Predicted half-life: {predicted_half_life}')


