import pickle
import torch
import numpy as np
from src.models import MULTModel
from modules.transformer import TransformerEncoder
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 读取数据
with open("/kaggle/working/unaligned_50.pkl", "rb") as f:
    all_data = pickle.load(f)

train_data = all_data['train']
valid_data = all_data['valid']
test_data = all_data['test']

train_data_audio = train_data['audio']
train_data_vision = train_data['vision']
train_data_audio_lengths = train_data['audio_lengths']
train_data_vision_lengths = train_data['vision_lengths']
valid_data_audio = valid_data['audio']
valid_data_vision = valid_data['vision']
valid_data_audio_lengths = valid_data['audio_lengths']
valid_data_vision_lengths = valid_data['vision_lengths']
test_data_audio = test_data['audio']
test_data_vision = test_data['vision']
test_data_audio_lengths = test_data['audio_lengths']
test_data_vision_lengths = test_data['vision_lengths']
train_data_audio = [item[:200, :] for item in train_data_audio]
valid_data_audio = [item[:200, :] for item in valid_data_audio]
test_data_audio = [item[:200, :] for item in test_data_audio]
train_data_vision = [item[:150, :] for item in train_data_vision]
valid_data_vision = [item[:150, :] for item in valid_data_vision]
test_data_vision = [item[:150, :] for item in test_data_vision]
train_data_audio_lengths = [min(num, 200) for num in train_data_audio_lengths]
valid_data_audio_lengths = [min(num, 200) for num in valid_data_audio_lengths]
test_data_audio_lengths = [min(num, 200) for num in test_data_audio_lengths]
train_data_vision_lengths = [min(num, 150) for num in train_data_vision_lengths]
valid_data_vision_lengths = [min(num, 150) for num in valid_data_vision_lengths]
test_data_vision_lengths = [min(num, 150) for num in test_data_vision_lengths]
train_data['audio'] = train_data_audio
valid_data['audio'] = valid_data_audio
test_data['audio'] = test_data_audio
train_data['vision'] = train_data_vision
valid_data['vision'] = valid_data_vision
test_data['vision'] = test_data_vision
train_data['audio_lengths'] = train_data_audio_lengths
valid_data['audio_lengths'] = valid_data_audio_lengths
test_data['audio_lengths'] = test_data_audio_lengths
train_data['vision_lengths'] = train_data_vision_lengths
valid_data['vision_lengths'] = valid_data_vision_lengths
test_data['vision_lengths'] = test_data_vision_lengths


class CustomDataset(Dataset):
    def __init__(self, data):
        self.visual_data = data['vision']
        self.audio_data = data['audio']
        self.label_data = data['regression_labels']
        self.text = data['raw_text']
        self.visual_lengths = data['vision_lengths']
        self.audio_lengths = data['audio_lengths']

    def __len__(self):
        return len(self.label_data)

    def __getitem__(self, idx):
        # print(idx)
        visual_feature = torch.tensor(self.visual_data[idx], dtype=torch.float32).clone().detach()
        visual_lengths = torch.tensor(self.visual_lengths[idx]).unsqueeze(-1).clone().detach()
        audio_feature = torch.tensor(self.audio_data[idx], dtype=torch.float32).clone().detach()
        audio_lengths = torch.tensor(self.audio_lengths[idx]).unsqueeze(-1).clone().detach()
        text = self.text[idx]
        # 数据增强
        noise = torch.randn_like(visual_feature) * 0.01
        visual_feature += noise
        noise = torch.randn_like(audio_feature) * 0.01
        audio_feature += noise
        label = torch.tensor(self.label_data[idx], dtype=torch.float32).clone().detach()

        return text, audio_feature, audio_lengths, visual_feature, visual_lengths, label


train_dataset = CustomDataset(train_data)
valid_dataset = CustomDataset(valid_data)
test_dataset = CustomDataset(test_data)

# 优化数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 网络搭建
class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        try:
            value = self[key]
            if isinstance(value, dict):
                value = DotDict(value)
            return value
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key)

    def __deepcopy__(self, memo):
        return DotDict(dict(self))


hyp_params={
    "orig_d_l" : 768,
    "orig_d_a" : 5,
    "orig_d_v" : 20,
    "vonly" : 0,
    "aonly" : 0,
    "lonly" : 0,
    "num_heads" : 6,
    "layers" : 6,
    "attn_dropout" : 0.2,
    "attn_dropout_a" : 0.2,
    "attn_dropout_v" : 0.2,
    "relu_dropout" : 0.2,
    "res_dropout" : 0.,
    "out_dropout" : 0.2,
    "embed_dropout" : 0.2,
    "attn_mask" : False,
    "output_dim" : 768,

}
args = DotDict(hyp_params)

## BERT Model
from transformers import BertTokenizer, BertModel
class Custom_Bert_Model(nn.Module):
    def __init__(self):
        super(Custom_Bert_Model, self).__init__()
        self.model_name = 'bert-base-uncased'
        self.bert_model = BertModel.from_pretrained(model_name, output_hidden_states=True)
        for param in self.bert_model.parameters():
            param.requires_grad = False
        for param in self.bert_model.encoder.layer[-2:].parameters():
            param.requires_grad = True



    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids, attention_mask=attention_mask)

        # 提取最终的隐藏层向量
        final_hidden_state = outputs.hidden_states[-1]  # 最后一个隐藏层 [batch_size, seq_length, hidden_size]

        # 提取 [CLS] 标记的隐藏状态
        cls_hidden_state = final_hidden_state[:, 0, :]  # 最后一个隐藏层，第一个 token（[CLS]） [batch_size, hidden_size]

        return final_hidden_state, cls_hidden_state


# 在 V_A_Embedding_Model 中增加卷积层 best
class V_A_Embedding_Model(nn.Module):
    def __init__(self, hyp_params):
        super(V_A_Embedding_Model, self).__init__()
        self.hidden_size = 128
        self.conv1d_audio = nn.Conv1d(hyp_params.orig_d_a, self.hidden_size, kernel_size=3, padding=1)
        self.conv1d_visual = nn.Conv1d(hyp_params.orig_d_v, self.hidden_size, kernel_size=3, padding=1)
        self.audio_embedding = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True,
                                       num_layers=2)
        self.visual_embedding = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True,
                                        num_layers=2)
        self.visual_embedding_linear = nn.Linear(self.hidden_size, hyp_params.output_dim)
        self.audio_embedding_linear = nn.Linear(self.hidden_size, hyp_params.output_dim)

    def forward(self, audio_x, length_audio_x, visual_x, length_visual_x):
        device = audio_x.device
        norm_visual_x = F.normalize(visual_x, dim=2)
        norm_audio_x = F.normalize(audio_x, dim=2)

        # 使用卷积层提取特征
        audio_x_conv = F.relu(self.conv1d_audio(norm_audio_x.permute(0, 2, 1)))
        visual_x_conv = F.relu(self.conv1d_visual(norm_visual_x.permute(0, 2, 1)))

        audio_x_conv = audio_x_conv.permute(0, 2, 1)
        visual_x_conv = visual_x_conv.permute(0, 2, 1)

        visual_outputs, (_, _) = self.visual_embedding(visual_x_conv)
        audio_outputs, (_, _) = self.audio_embedding(audio_x_conv)

        visual_embedding_hidden = visual_outputs[:, -1, :]
        audio_embedding_hidden = audio_outputs[:, -1, :]

        visual_embedding_tensor = F.relu(self.visual_embedding_linear(F.relu(visual_embedding_hidden)))
        audio_embedding_tensor = F.relu(self.audio_embedding_linear(F.relu(audio_embedding_hidden)))

        return audio_embedding_tensor, audio_outputs, visual_embedding_tensor, visual_outputs



# 增强层
class Aug_Model(nn.Module):
    def __init__(self, hyp_params):
        super(Aug_Model, self).__init__()
        self.args = deepcopy(hyp_params)
        self.args.orig_d_a = 128
        self.args.orig_d_v = 128
        self.text_args = deepcopy(self.args)
        self.text_args.lonly = 1
        self.text_aug_model = MULTModel(self.text_args)
        self.visual_args = deepcopy(self.args)
        self.visual_args.aonly = 1
        self.visual_aug_model = MULTModel(self.visual_args)
        self.audio_args = deepcopy(self.args)
        self.audio_args.vonly = 1

        self.audio_aug_model = MULTModel(self.audio_args)
    def forward(self, text, audio, visual):
        text_aug = self.text_aug_model(text, audio, visual)[0]
        visual_aug = self.visual_aug_model(text, audio, visual)[0]
        audio_aug = self.audio_aug_model(text, audio, visual)[0]
        return text_aug, audio_aug, visual_aug


# 选择层 1536 best
class TWD_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.reject_selection_layer = nn.Linear(1536, 1536)
        self.reconsiderate_selection_layer = nn.Linear(1536, 1536)
        self.reject_visual_batch_norm = nn.BatchNorm1d(1536)
        self.reject_audio_batch_norm = nn.BatchNorm1d(1536)
        self.recon_visual_batch_norm = nn.BatchNorm1d(1536)
        self.recon_audio_batch_norm = nn.BatchNorm1d(1536)

    def forward(self, text_embedding, audio_embedding, visual_embedding):
        # Reject Seletion
        text_selection_mask_score = F.relu(self.reject_selection_layer(text_embedding))
        text_selection_mask = torch.where(text_selection_mask_score > 0, torch.tensor(1), torch.tensor(0))
        text_selection = torch.mul(text_embedding, text_selection_mask)
        text_rejection_loss = 1 - F.cosine_similarity(text_embedding, text_selection)

        visual_selection_mask_score = F.relu(self.reject_selection_layer(self.reject_visual_batch_norm(visual_embedding)))
        #visual_selection_mask = F.tanh(self.reject_selection_layer(self.reject_visual_batch_norm(visual_embedding))
        visual_selection_mask = torch.where(visual_selection_mask_score > 0, torch.tensor(1), torch.tensor(0))
        visual_selection = torch.mul(visual_embedding, visual_selection_mask)
        visual_rejection_loss = 1 - F.cosine_similarity(visual_embedding, visual_selection)

        audio_selection_mask_score = F.relu(self.reject_selection_layer(self.reject_audio_batch_norm(audio_embedding)))
        audio_selection_mask = torch.where(audio_selection_mask_score > 0, torch.tensor(1), torch.tensor(0))
        audio_selection = torch.mul(audio_embedding, audio_selection_mask)
        audio_rejection_loss = 1 - F.cosine_similarity(audio_embedding, audio_selection)

        # Reconsiderate Selection
        text_reconsiderate_mask_score = F.relu(self.reconsiderate_selection_layer(text_embedding))
        text_reconsiderate_mask = torch.where(text_reconsiderate_mask_score > 0, torch.tensor(1), torch.tensor(0))
        text_accept = torch.mul(text_reconsiderate_mask, text_selection)
        text_accept_loss = 1 - F.cosine_similarity(text_embedding, text_accept) - text_rejection_loss

        visual_reconsiderate_mask_score = F.relu(self.reconsiderate_selection_layer(self.recon_visual_batch_norm(visual_selection)))
        visual_reconsiderate_mask = torch.where(visual_reconsiderate_mask_score > 0, torch.tensor(1), torch.tensor(0))
        visual_accept = torch.mul(visual_reconsiderate_mask, visual_selection)
        visual_accept_loss = 1 - F.cosine_similarity(visual_embedding, visual_accept) - visual_rejection_loss

        audio_reconsiderata_mask_score = F.relu(self.reconsiderate_selection_layer(self.recon_audio_batch_norm(audio_selection)))
        audio_reconsiderata_mask = torch.where(audio_reconsiderata_mask_score > 0, torch.tensor(1), torch.tensor(0))
        audio_accept = torch.mul(audio_reconsiderata_mask, audio_selection)
        audio_accept_loss = 1 - F.cosine_similarity(audio_embedding, audio_accept) - audio_rejection_loss

        # loss matrix
        text_loss = torch.stack((1 - text_rejection_loss - text_accept_loss, text_accept_loss, text_rejection_loss), dim=1)
        visual_loss = torch.stack((1 - visual_rejection_loss - visual_accept_loss, visual_accept_loss, visual_rejection_loss), dim=1)
        audio_loss = torch.stack((1 - audio_rejection_loss - audio_accept_loss, audio_accept_loss, audio_rejection_loss), dim=1)

        return text_accept, text_selection - text_accept, text_loss, audio_accept, audio_selection - audio_accept, audio_loss, visual_accept, visual_selection - visual_accept, visual_loss


# 选择注意力层 Xavier初始化 best
class TWD_Attention_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_encoder = TransformerEncoder(embed_dim=1536,
                                      num_heads=8,
                                      layers=2,
                                      attn_dropout=0.2,
                                      relu_dropout=0.2,
                                      res_dropout=0.,
                                      embed_dropout=0.2,
                                      attn_mask=False,
                                      pos_encoding=False,
                                      )
        #self.mlp = nn.Linear(768*6,1)
        self.final_layer = TransformerEncoder(embed_dim=1536,
                                      num_heads=8,
                                      layers=2,
                                      attn_dropout=0.2,
                                      relu_dropout=0.2,
                                      res_dropout=0.,
                                      embed_dropout=0.2,
                                      attn_mask=False,
                                      pos_encoding=False,
                                     )


    def forward(self, text_accept, text_reconsiderate, audio_accept, audio_reconsiderate, visual_accept, visual_reconsiderate):
        text_accept_norm = text_accept.unsqueeze(1)
        visual_accept_norm = visual_accept.unsqueeze(1)
        audio_accept_norm = audio_accept.unsqueeze(1)
        concat = torch.cat((text_accept_norm,visual_accept_norm,audio_accept_norm), dim=1).permute(1, 0, 2)
        output_norm =self.transformer_encoder(concat).permute(1, 0, 2)
        output_norm[:,0,:] = output_norm[:,0,:] + text_reconsiderate
        output_norm[:,1,:] = output_norm[:,1,:] + visual_reconsiderate
        output_norm[:,2,:] = output_norm[:,2,:] + audio_reconsiderate
        output = F.relu(output_norm.permute(1, 0, 2))
        final_output = self.final_layer(output)

        return final_output.permute(1, 0, 2)


# best
class TWD_ATTN_Model_Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.selection = TWD_Model()
        self.attention = TWD_Attention_Model()
        self.fc1 = nn.Linear(1536 * 3, 1536)  # 确保全连接层在初始化时定义

    def forward(self, text_embedding, audio_embedding, visual_embedding):
        text_accept, text_re, text_loss, audio_accept, audio_re, audio_loss, visual_accept, visual_re, visual_loss = self.selection(
            text_embedding, audio_embedding, visual_embedding)

        loss_matrix = (text_loss.mean(dim=0) + audio_loss.mean(dim=0) + visual_loss.mean(dim=0)) / 3

        concat = self.attention(text_accept, text_re, audio_accept, audio_re, visual_accept, visual_re)
        text_mid, audio_mid, visual_mid = torch.split(concat, 1, dim=1)

        return text_mid.squeeze(1), audio_mid.squeeze(1), visual_mid.squeeze(1), loss_matrix


class TWD_ATTN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = TWD_ATTN_Model_Layer()
        self.layer2 = TWD_ATTN_Model_Layer()
        self.fc1 = nn.Linear(4608, 768)  # 新增全连接层
        self.fc2 = nn.Linear(768, 1)

    def forward(self, text_embedding, audio_embedding, visual_embedding):
        text_1, audio_1, visual_1, matrix_1 = self.layer1(F.normalize(text_embedding, dim=1),
                                                          F.normalize(audio_embedding, dim=1),
                                                          F.normalize(visual_embedding, dim=1))
        text_2, audio_2, visual_2, matrix_2 = self.layer2(F.normalize(text_1, dim=1),
                                                          F.normalize(audio_1, dim=1),
                                                          F.normalize(visual_1, dim=1))
        combined = torch.cat((text_2, audio_2, visual_2), dim=1)
        combined = F.relu(self.fc1(combined))
        output = self.fc2(combined)

        return output, (matrix_1 + matrix_2) / 2


# 训练指标
class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2,2)
        summed = torch.sum(power)
        sqrt = summed**(0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)
class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, a, b):
        # 计算余弦相似度
        cos_sim = 1-F.cosine_similarity(a, b, dim=1)
        # 将余弦相似度转换为损失，因为我们想要最大化余弦相似度，所以取其负值
        loss = torch.mean(cos_sim)
        return loss

def compute_squared_frobenius_norm(matrix):
    # 复制输入矩阵，以避免修改原始矩阵
    matrix_copy = matrix.clone()

    # Step 1: 减去所有元素的均值
    mean_value = torch.mean(matrix_copy)
    matrix_copy -= mean_value

    # Step 2: 将矩阵展平为向量并计算其 L2 范数
    l2_norm = torch.norm(matrix_copy)

    # 计算 squared Frobenius norm
    squared_frobenius_norm = torch.pow(l2_norm, 2)

    # 规范化 squared Frobenius norm
    normalized_squared_frobenius_norm = squared_frobenius_norm / (matrix.shape[0] * matrix.shape[1])

    return normalized_squared_frobenius_norm

def a2_compute(outputs, labels):
    outputs = torch.where(outputs >= 0, torch.tensor(1), torch.tensor(0))
    labels = torch.where(labels >= 0, torch.tensor(1), torch.tensor(0))
    TP = torch.sum((outputs == 1) & (labels == 1))
    TN = torch.sum((outputs == 0) & (labels == 0))
    FP = torch.sum((outputs == 1) & (labels == 0))
    FN = torch.sum((outputs == 0) & (labels == 1))
    return TP+TN, torch.Tensor([TP,TN,FP,FN])


def a7_compute(outputs, labels):
    round_outputs = torch.round(outputs)
    round_labels = torch.round(labels)
    return torch.sum(round_outputs == round_labels)

def clip_tensor(tensor):
    # 创建一个与原始张量相同大小的新张量
    clipped_tensor = torch.empty_like(tensor)

    # 对第二个维度上的第一个值进行裁剪到[-3,+3]
    clipped_tensor[:, 0] = torch.clamp(tensor[:, 0], -3.0, 3.0)

    # 对其他维度上的值进行裁剪到[0, 3.0]
    clipped_tensor[:, 1:] = torch.clamp(tensor[:, 1:], 0.0, 3.0)

    return clipped_tensor



# 创建模型实例 BERT
from transformers import BertTokenizer, AutoTokenizer, AutoConfig, AutoModel
model_name = 'bert-base-uncased'
# 创建模型实例
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练的BERT模型和分词器
bert_tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = Custom_Bert_Model().to(device)
embedding_model = V_A_Embedding_Model(args).to(device)
aug_model =  Aug_Model(args).to(device)
twd_model = TWD_ATTN_Model().to(device)
# 定义损失函数和优化器
criterion = nn.L1Loss()
criterion_sim = CosineSimilarityLoss()
optimizer = torch.optim.Adam([
    {'params': filter(lambda p: p.requires_grad, bert_model.parameters()), 'lr': 2e-5, 'weight_decay': 1e-4},
    #{'params': bert_model.parameters(), 'lr': 2e-5, 'weight_decay': 1e-4},  # 增加L2正则化
    {'params': embedding_model.parameters(), 'lr': 5e-5, 'weight_decay': 1e-4},
    {'params': aug_model.parameters(), 'lr': 5e-5, 'weight_decay': 1e-4},
    {'params': twd_model.parameters(), 'lr': 5e-5, 'weight_decay': 1e-4},
])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)


from tqdm import tqdm_notebook
# 定义训练函数 代价敏感
def train(bert_model, bert_tokenizer, embedding_model, aug_model, twd_model, train_loader, criterion, criterion_sim, optimizer, device):
    bert_model.train()
    embedding_model.train()
    aug_model.train()
    twd_model.train()
    mae_loss = 0.0
    custom_loss = 0.0
    for text, audio_feature, audio_lengths, visual_feature, visual_lengths, label in tqdm_notebook(train_loader, total=len(train_loader)):
        audio_feature, audio_lengths, visual_feature, visual_lengths, label \
        = audio_feature.to(device), audio_lengths.to(device), visual_feature.to(device), visual_lengths.to(device), label.to(device)

        optimizer.zero_grad()

        # embedding model
        inputs = bert_tokenizer(text, padding='max_length', truncation=True, max_length=50, return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        text_outputs, text_hidden_embedding = bert_model(input_ids, attention_mask)
        ################
#         assert torch.isnan(text_outputs).sum() == 0, print(text_outputs)
#         assert torch.isnan(text_hidden_embedding).sum() == 0, print(text_hidden_embedding)
        #########################
        audio_embedding, audio_outputs, visual_embedding, visual_outputs = embedding_model(audio_feature, audio_lengths, visual_feature, visual_lengths)
        ###################################
#         assert torch.isnan(audio_embedding).sum() == 0, print(audio_embedding)
#         assert torch.isnan(audio_outputs).sum() == 0, print(audio_outputs)
#         assert torch.isnan(visual_embedding).sum() == 0, print(visual_embedding)
#         assert torch.isnan(visual_outputs).sum() == 0, print(visual_outputs)
        #####################################
        length = visual_embedding.shape[0]
        # compute similarity loss
        visual_sim_loss = criterion_sim(visual_embedding, text_hidden_embedding)
        audio_sim_loss = criterion_sim(audio_embedding, text_hidden_embedding)
        sim_loss = (visual_sim_loss + audio_sim_loss)*0.5
        #sim_loss.backward()

        # aug model
        text_aug, audio_aug, visual_aug = aug_model(text_outputs, audio_outputs, visual_outputs)

        # concat the tensor in need
        text_concat = torch.cat((text_hidden_embedding.detach(), text_aug), dim=1)
        audio_concat = torch.cat((audio_embedding.detach(), audio_aug), dim=1)
        visual_concat = torch.cat((visual_embedding.detach(), audio_aug), dim=1)

        # twd_model
        outputs, loss_matrix = twd_model(text_concat, audio_concat, visual_concat)

        # clamped output
        outputs_clamped = clip_tensor(outputs)

        # compute mae_loss
        mae_loss_item = criterion(outputs_clamped, label.unsqueeze(-1))

        mul = torch.Tensor([1.2,1.,1.2]).to(device)
        alpha = torch.sum(torch.mul(mul, loss_matrix))
        custom_mae_loss_item  = alpha * mae_loss_item
        # Total loss
        custom_loss_item = sim_loss + custom_mae_loss_item
        custom_loss_item.backward()

        #####排查loss#####
#         assert torch.isnan(custom_loss_item).sum() == 0, print(custom_loss_item)


        # with torch.autograd.detect_anomaly():

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(parameters=embedding_model.parameters(), max_norm=1)
        torch.nn.utils.clip_grad_norm_(parameters=aug_model.parameters(), max_norm=1)
        torch.nn.utils.clip_grad_norm_(parameters=twd_model.parameters(), max_norm=1)

        ###step前排查模型参数########
#         for i,param in enumerate(embedding_model.parameters()):
#             if torch.isnan(param).any():
#                 print("step前--------------------------------------")
#                 print(f"embedding_model{i}{param}")
#         for i,param in enumerate(aug_model.parameters()):
#             if torch.isnan(param).any():
#                 print("step前--------------------------------------")
#                 print(f"aug_model{i}{param}")
#         for i,param in enumerate(twd_model.parameters()):
#             if torch.isnan(param).any():
#                 print("step前--------------------------------------")
#                 print(f"twd_model{i}{param}")


        optimizer.step()

        ########step后排查模型参数########
        for id,param in enumerate(embedding_model.parameters()):
            if torch.isnan(param).any():
                print("step后--------------------------------------")
                print(f"embedding_model_{id}{param}")
        for id,param in enumerate(aug_model.parameters()):
            if torch.isnan(param).any():
                print("step后--------------------------------------")
                print(f"aug_model_{id}{param}")
        for id,param in enumerate(twd_model.parameters()):
            if torch.isnan(param).any():
                print("step后--------------------------------------")
                print(f"twd_model_{id}{param}")


        mae_loss += mae_loss_item.item()
        custom_loss += custom_loss_item.item()


    return mae_loss / len(train_loader) , custom_loss / len(train_loader)


# 定义训练函数 代价敏感
def valid(bert_model, bert_tokenizer, embedding_model, aug_model, twd_model, valid_loader, criterion, criterion_sim, device, test=False):
    bert_model.eval()
    embedding_model.eval()
    aug_model.eval()
    twd_model.eval()
    mae_loss = 0.0
    custom_loss = 0.0
    a2 = 0.0
    a7 = 0.0
    f1_ele = torch.Tensor([0.,0.,0.,0.])
    y_label = []
    y_pred = []
    global best_mae
    with torch.no_grad():
        for text, audio_feature, audio_lengths, visual_feature, visual_lengths, label in tqdm_notebook(valid_loader, total=len(valid_loader)):
            audio_feature, audio_lengths, visual_feature, visual_lengths, label \
            = audio_feature.to(device), audio_lengths.to(device), visual_feature.to(device), visual_lengths.to(device), label.to(device)


            # embedding model
            inputs = bert_tokenizer(text, padding='max_length', truncation=True, max_length=50, return_tensors='pt')
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            text_outputs, text_hidden_embedding = bert_model(input_ids, attention_mask)
            audio_embedding, audio_outputs, visual_embedding, visual_outputs = embedding_model(audio_feature, audio_lengths, visual_feature, visual_lengths)
            length = visual_embedding.shape[0]
            # compute similarity loss
            visual_sim_loss = criterion_sim(visual_embedding, text_hidden_embedding)
            audio_sim_loss = criterion_sim(audio_embedding, text_hidden_embedding)
            sim_loss = (visual_sim_loss + audio_sim_loss)*0.5


            # aug model
            text_aug, audio_aug, visual_aug = aug_model(text_outputs.detach(), audio_outputs.detach(), visual_outputs.detach())

            # concat the tensor in need
            text_concat = torch.cat((text_hidden_embedding.detach(), text_aug), dim=1)
            audio_concat = torch.cat((audio_embedding.detach(), audio_aug), dim=1)
            visual_concat = torch.cat((visual_embedding.detach(), audio_aug), dim=1)

            # twd_model
            outputs, loss_matrix = twd_model(text_concat, audio_concat, visual_concat)
            # clamped output
            outputs_clamped = clip_tensor(outputs)
            # compute mae_loss
            mae_loss_item = criterion(outputs_clamped, label.unsqueeze(-1))

            mul = torch.Tensor([1.2,1.,1.2]).to(device)
            alpha = torch.sum(torch.mul(mul, loss_matrix))
            custom_mae_loss_item  = alpha * mae_loss_item
            # Total loss
            custom_loss_item = sim_loss + custom_mae_loss_item



            # predict result
            y_pred += outputs_clamped.squeeze().tolist()
            y_label += label.tolist()

            # compute related metrics
            a2_item, f1_ele_item = a2_compute(outputs_clamped, label.unsqueeze(-1))
            a7_item = a7_compute(outputs_clamped, label.unsqueeze(-1))
            a7 += a7_item
            a2 += a2_item
            f1_ele = torch.add(f1_ele,f1_ele_item)





            mae_loss += mae_loss_item.item()
            custom_loss += custom_loss_item.item()
      # compute f1-score
        total_precision = f1_ele[0]/(f1_ele[0]+f1_ele[2])
        total_recall = f1_ele[0]/(f1_ele[0]+f1_ele[3])
        total_f1 = 2 * (total_precision * total_recall) / (total_precision + total_recall)

        corr = torch.corrcoef(torch.stack((torch.tensor(y_pred), torch.tensor(y_label))))[0][1]

        if (best_mae > mae_loss / len(valid_loader)) & test:
            best_mae = mae_loss / len(valid_loader)
            print("hello")
            torch.save(embedding_model.state_dict(), '/kaggle/working/best_embedding_model.pth')
            torch.save(aug_model.state_dict(), '/kaggle/working/best_aug_model.pth')
            torch.save(twd_model.state_dict(), '/kaggle/working/best_twd_model.pth')
            torch.save(bert_model.state_dict(), '/kaggle/working/best_bert_model.pth')


    return mae_loss / len(valid_loader) , custom_loss / len(valid_loader), a2 / len(valid_loader.dataset), total_f1, a7 / len(valid_loader.dataset), corr


# no.1 with 1.2 bert fine tune  last hidden with  lr=1e-4 improved embedding  2-layer
num_epochs = 100
best_mae = 0.77
for epoch in range(num_epochs):
    train_mae_loss, train_custom_loss = train(bert_model, bert_tokenizer, embedding_model, aug_model, twd_model,  train_loader, criterion, criterion_sim, optimizer, device)
    valid_mae_loss, valid_custom_loss, valid_a2, valid_f1, valid_a7, valid_corr = valid(bert_model, bert_tokenizer, embedding_model, aug_model, twd_model,  valid_loader, criterion, criterion_sim, device)
    test_mae_loss, test_custom_loss, test_a2, test_f1, test_a7, test_corr = valid(bert_model, bert_tokenizer, embedding_model, aug_model, twd_model, test_loader, criterion, criterion_sim, device, test=True)
    print(f'Epoch {epoch+1}/{num_epochs}, Train_mae_Loss: {train_mae_loss:.4f}, Train_custom_Loss: {train_custom_loss}\
       valid_mae_Loss: {valid_mae_loss:.4f}, valid_custom_Loss: {valid_custom_loss:.4f},\
       test_mae_Loss:{test_mae_loss:.4f}, test_custom_Loss:{test_custom_loss:.4f}')
    print(f'Epoch {epoch+1}/{num_epochs}, Valid_a2:{valid_a2:.4f}, Test_a2:{test_a2:.4f}')
    print(f'Epoch {epoch+1}/{num_epochs}, Valid_f1:{valid_f1:.4f}, Test_f1:{test_f1:.4f}')
    print(f'Epoch {epoch+1}/{num_epochs}, Valid_a7:{valid_a7:.4f}, Test_a7:{test_a7:.4f}')
    print(f'Epoch {epoch+1}/{num_epochs}, Valid_corr:{valid_corr:.4f}, Test_corr:{test_corr:.4f}')