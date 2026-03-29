import numpy as np
import collections
import torch
import torch.optim as optim

import rnn

start_token = 'G'
end_token = 'E'
batch_size = 64


def process_poems1(file_name):
    poems = []
    with open(file_name, "r", encoding='utf-8', ) as f:
        for line in f.readlines():
            try:
                title, content = line.strip().split(':')
                content = content.replace(' ', '')
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                                start_token in content or end_token in content:
                    continue
                if len(content) < 5 or len(content) > 80:
                    continue
                content = start_token + content + end_token
                poems.append(content)
            except ValueError as e:
                pass  # 删掉 print("error")
    poems = sorted(poems, key=lambda line: len(line))
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)
    words = words[:len(words)] + (' ',)
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]
    return poems_vector, word_int_map, words


def generate_batch(batch_size, poems_vec, word_to_int):
    n_chunk = len(poems_vec) // batch_size
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size
        x_data = poems_vec[start_index:end_index]
        y_data = []
        for row in x_data:
            y  = row[1:]
            y.append(row[-1])
            y_data.append(y)
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches


def run_training():
    poems_vector, word_to_int, vocabularies = process_poems1('./poems.txt')
    print("finish loading data")
    BATCH_SIZE = 64

    torch.manual_seed(5)
    word_embedding = rnn.word_embedding(vocab_length= len(word_to_int) + 1 , embedding_dim= 100)
    rnn_model = rnn.RNN_model(batch_sz = BATCH_SIZE,
                              vocab_len = len(word_to_int) + 1,
                              word_embedding = word_embedding,
                              embedding_dim= 100,
                              lstm_hidden_dim=128)

    optimizer = optim.RMSprop(rnn_model.parameters(), lr=0.001)
    loss_fun = torch.nn.NLLLoss()

    for epoch in range(30):
        batches_inputs, batches_outputs = generate_batch(BATCH_SIZE, poems_vector, word_to_int)
        n_chunk = len(batches_inputs)
        epoch_loss = 0

        for batch in range(n_chunk):
            batch_x = batches_inputs[batch]
            batch_y = batches_outputs[batch]

            batch_loss = 0
            for index in range(BATCH_SIZE):
                x = np.array(batch_x[index], dtype=np.int64)
                y = np.array(batch_y[index], dtype=np.int64)
                x = torch.from_numpy(x).unsqueeze(1)
                y = torch.from_numpy(y)
                
                pre = rnn_model(x)
                batch_loss += loss_fun(pre, y)

                if batch == 0 and index == 0:
                    _, pre_idx = torch.max(pre, dim=1)
                    print(f'\nEpoch {epoch} 示例预测')
                    print('预测:', pre_idx.data.tolist())
                    print('目标:', y.data.tolist())

            batch_loss = batch_loss / BATCH_SIZE
            epoch_loss += batch_loss.item()

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), 1)
            optimizer.step()

            if batch % 20 == 0:
                torch.save(rnn_model.state_dict(), './poem_generator_rnn')

        avg_epoch_loss = epoch_loss / n_chunk
        print(f"Epoch {epoch:2d} 平均损失: {avg_epoch_loss:.4f}")


def to_word(predict, vocabs):
    sample = np.argmax(predict)
    if sample >= len(vocabs):
        sample = len(vocabs) - 1
    return vocabs[sample]


def pretty_print_poem(poem):
    shige=[]
    for w in poem:
        if w == start_token or w == end_token:
            break
        shige.append(w)
    poem_sentences = poem.split('。')
    for s in poem_sentences:
        if s != '' and len(s) > 2:
            print(s + '。')


def gen_poem(begin_word):
    poem = start_token + begin_word
    word = begin_word
    for _ in range(40):
        input_idxs = [word_int_map.get(w, word_int_map[' ']) for w in poem]
        input_tensor = torch.from_numpy(np.array(input_idxs, dtype=np.int64)).unsqueeze(1)
        output = rnn_model(input_tensor, is_test=True)
        word = to_word(output.data.tolist()[-1], vocabularies)
        if word == end_token:
            break
        poem += word
    return poem[1:]


# ==================== 运行 ====================
run_training()  # 训练

# 加载模型生成诗歌
poems_vector, word_int_map, vocabularies = process_poems1('./poems.txt')
vocab_len = len(word_int_map) + 1

word_embedding = rnn.word_embedding(vocab_length=vocab_len, embedding_dim=100)
rnn_model = rnn.RNN_model(batch_sz=64,
                         vocab_len=vocab_len,
                         word_embedding=word_embedding,
                         embedding_dim=100,
                         lstm_hidden_dim=128)

rnn_model.load_state_dict(torch.load('./poem_generator_rnn'))
rnn_model.eval()

keywords = ["日", "红", "山", "夜", "湖", "君", "海", "月"]
for kw in keywords:
    result = gen_poem(kw)
    print(f"\n【{kw}】开头：")
    pretty_print_poem(result)
    print('-'*30)