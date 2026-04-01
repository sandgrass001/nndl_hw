import numpy as np
import collections
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# ======================
# 基本配置
# ======================
start_token = 'G'
end_token = 'E'
BATCH_SIZE = 64
EPOCHS = 30
LR = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================
# 数据集
# ======================
class PoemDataset(Dataset):
    def __init__(self, file_name):
        self.poems = []

        with open(file_name, "r", encoding='utf-8') as f:
            for line in f:
                try:
                    title, content = line.strip().split(':')
                    content = content.replace(' ', '')

                    if any(c in content for c in ['_', '(', '（', '《', '[']):
                        continue
                    if len(content) < 10 or len(content) > 80:
                        continue

                    content = start_token + content + end_token
                    self.poems.append(content)
                except:
                    continue

        # 构建词表
        all_words = []
        for poem in self.poems:
            all_words += list(poem)

        counter = collections.Counter(all_words)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])

        self.words, _ = zip(*count_pairs)
        self.words = list(self.words) + [' ']

        self.word2idx = {w: i for i, w in enumerate(self.words)}
        self.idx2word = dict(enumerate(self.words))

        # 转 index
        self.data = [
            [self.word2idx.get(w, self.word2idx[' ']) for w in poem]
            for poem in self.poems
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]


# ======================
# padding
# ======================
def collate_fn(batch):
    x, y = zip(*batch)

    max_len = max(len(seq) for seq in x)

    def pad(seq):
        return seq + [0] * (max_len - len(seq))

    x = [pad(seq) for seq in x]
    y = [pad(seq) for seq in y]

    return torch.LongTensor(x), torch.LongTensor(y)


# ======================
# 模型
# ======================
class PoemModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=2,
            dropout=0.3,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out


# ======================
# 训练（已添加：预测 vs 目标）
# ======================
def train():
    dataset = PoemDataset('./poems.txt')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = PoemModel(len(dataset.words)).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(EPOCHS):
        total_loss = 0

        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)

            # 计算损失
            out_reshaped = out.reshape(-1, out.shape[-1])
            y_reshaped = y.reshape(-1)
            loss = loss_fn(out_reshaped, y_reshaped)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            # ============ 【关键：每轮打印 预测 vs 目标】 ============
            if batch_idx == 0:
                # 取第一条数据查看
                sample_out = out[0]
                sample_y = y[0]

                pred_idx = sample_out.argmax(dim=-1).cpu().numpy().tolist()
                true_idx = sample_y.cpu().numpy().tolist()

                print(f"\n=== Epoch {epoch} 示例预测 ===")
                print(f"预测: {pred_idx[:10]}")  # 只打印前10个，避免太长
                print(f"目标: {true_idx[:10]}")
                print("-" * 40)
            # ======================================================

        print(f"Epoch {epoch:2d} 平均Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "poem_model.pt")
    return model, dataset


# ======================
# 生成
# ======================
def generate(model, dataset, start_word, max_len=40, temperature=0.8):
    model.eval()

    word = start_word
    result = start_word
    input_idx = torch.LongTensor([[dataset.word2idx.get(word, 0)]]).to(DEVICE)

    for _ in range(max_len):
        out = model(input_idx)
        logits = out[:, -1, :] / temperature
        prob = F.softmax(logits, dim=-1)
        idx = torch.multinomial(prob, 1).item()
        word = dataset.idx2word[idx]

        if word == end_token:
            break

        result += word
        input_idx = torch.LongTensor([[idx]]).to(DEVICE)

    return result


# ======================
# 主程序
# ======================
if __name__ == "__main__":
    model, dataset = train()

    keywords = ["日", "山", "夜", "月", "风"]
    for kw in keywords:
        poem = generate(model, dataset, kw, temperature=0.7)
        print(f"\n【{kw}】开头的诗：")
        print(poem)
        print("-" * 30)