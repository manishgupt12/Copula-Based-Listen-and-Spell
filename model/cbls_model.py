#Author: Manish Gupta (Syracuse University)
# 	: Saikiran Bulusu (Syracuse University)
#	: Shan Zhang (Syracuse University)

#Source: Kaituo XU (Northwestern Polytechnical University)
#Refernce	: William Chan (Carnegie Mellon University)
#			: Navdeep Jaitly (Google Brain)
#			: Quoc Le (Google Brain)
# 			: Oriol Vinyal(Google Brain)
# 	Chan, William, et al. "Listen, attend and spell: A neural network for large vocabulary conversational speech recognition." 
#	Acoustics, Speech and Signal Processing (ICASSP), 2016 IEEE International Conference on. IEEE, 2016.



import torch
if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device
from torch.autograd import Variable    
import torch.nn as nn
import torch.nn.functional as F

from util.functions import TimeDistributed,CreateOnehotVariable,Isanycolumnzero
import numpy as np
from scipy.stats import norm
from util.copulapdf import  copulapdf



# BLSTM layer for pBLSTM
# Step 1. Reduce time resolution to half
# Step 2. Run through BLSTM
# Note the input should have timestep%2 == 0
class pBLSTMLayer(nn.Module):
    def __init__(self,input_feature_dim,hidden_dim,rnn_unit='LSTM',dropout_rate=0.0):
        super(pBLSTMLayer, self).__init__()
        self.rnn_unit = getattr(nn,rnn_unit.upper())

        # feature dimension will be doubled since time resolution reduction
        self.BLSTM = self.rnn_unit(input_feature_dim*2,hidden_dim,1, bidirectional=True, 
                                   dropout=dropout_rate,batch_first=True)
    
    def forward(self,input_x):
        batch_size = input_x.size(0)
        timestep = input_x.size(1)
        feature_dim = input_x.size(2)
        # Reduce time resolution
        input_x = input_x.contiguous().view(batch_size,int(timestep/2),feature_dim*2)
        # Bidirectional RNN
        output,hidden = self.BLSTM(input_x)
        return output,hidden

# Listener is a pBLSTM stacking 3 layers to reduce time resolution 8 times
# Input shape should be [# of sample, timestep, features]
class Listener(nn.Module):
    def __init__(self, input_feature_dim, listener_hidden_dim, rnn_unit, use_gpu, dropout_rate=0.0, **kwargs):
        super(Listener, self).__init__()
        # Listener RNN layer
        self.pLSTM_layer1 = pBLSTMLayer(input_feature_dim,listener_hidden_dim, rnn_unit=rnn_unit, dropout_rate=dropout_rate)
        self.pLSTM_layer2 = pBLSTMLayer(listener_hidden_dim*2,listener_hidden_dim, rnn_unit=rnn_unit, dropout_rate=dropout_rate)
        #self.pLSTM_layer3 = pBLSTMLayer(listener_hidden_dim*2,listener_hidden_dim, rnn_unit=rnn_unit, dropout_rate=dropout_rate)
        self.use_gpu = use_gpu
        if self.use_gpu:
            self = self.cuda()

    def forward(self,input_x):
        #print("Input  ***********", np.shape(input_x))
        output, _ = self.pLSTM_layer1(input_x)
        #print("OUtput1 ***********", np.shape(output))
        output, _ = self.pLSTM_layer2(output)
        #print("OUtput2 ***********", np.shape(output))
        #output, _ = self.pLSTM_layer3(output)
        #print("OUtput3 ***********", np.shape(output))
        return output

#speller(listner_feature,count_original, count_copula,ground_truth=None,teacher_force_rate=0)
# Speller specified in the paper
class Speller(nn.Module):
    def __init__(self, output_class_dim,  speller_hidden_dim, rnn_unit, speller_rnn_layer, use_gpu, max_label_len,
                 use_mlp_in_attention, mlp_dim_in_attention, mlp_activate_in_attention, listener_hidden_dim, count_original,count_copula,**kwargs):
        super(Speller, self).__init__()
        self.rnn_unit = getattr(nn,rnn_unit.upper())
        self.max_label_len = max_label_len
        self.use_gpu = use_gpu
        self.float_type = torch.torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
        self.label_dim = output_class_dim
        self.count_original = count_original
        self.count_copula = count_copula
        #print("********************",count_copula, count_original)
        self.rnn_layer = self.rnn_unit(output_class_dim+speller_hidden_dim,speller_hidden_dim,num_layers=speller_rnn_layer)
        self.attention = Attention( mlp_preprocess_input=use_mlp_in_attention, preprocess_mlp_dim=mlp_dim_in_attention,use_gpu=use_gpu,
                                    activate=mlp_activate_in_attention, input_feature_dim=2*listener_hidden_dim)
        self.character_distribution = nn.Linear(speller_hidden_dim*2,output_class_dim)
        self.softmax = nn.LogSoftmax(dim=-1)
        if self.use_gpu:
            self = self.cuda()

    # Stepwise operation of each sequence
    def forward_step(self,input_word, last_hidden_state,listener_feature):
        rnn_output, hidden_state = self.rnn_layer(input_word,last_hidden_state)
        attention_score, context,count_orignal,count_copula = self.attention(rnn_output,listener_feature)
        concat_feature = torch.cat([rnn_output.squeeze(dim=1),context],dim=-1)
        raw_pred = self.softmax(self.character_distribution(concat_feature))
        self.count_copula +=count_copula
        self.count_original += count_orignal
        #print("*************far*******",self.count_copula, count_orignal)


        return raw_pred, hidden_state, context, attention_score

    def forward(self, listener_feature,ground_truth=None, teacher_force_rate = 0.9):
        if ground_truth is None:
            teacher_force_rate = 0
        teacher_force = True if np.random.random_sample() < teacher_force_rate else False
        #print("*****************for***", self.count_copula)
        batch_size = listener_feature.size()[0]
        output_word = CreateOnehotVariable(self.float_type(np.zeros((batch_size,1))),self.label_dim)
        rnn_input = torch.cat([output_word,listener_feature[:,0:1,:]],dim=-1)

        hidden_state = None
        raw_pred_seq = []
        output_seq = []
        attention_record = []
        for step in range(self.max_label_len):
            raw_pred, hidden_state, context, attention_score = self.forward_step(rnn_input, hidden_state, listener_feature)
            raw_pred_seq.append(raw_pred)
            attention_record.append(attention_score)
            # Teacher force - use ground truth as next step's input
            if teacher_force:
                output_word = ground_truth[:,step:step+1,:].type(self.float_type)
            else:
                output_word = raw_pred.unsqueeze(1)
            rnn_input = torch.cat([output_word,context.unsqueeze(1)],dim=-1)
        #print("********************",self.count_copula, self.count_original)

        return raw_pred_seq,attention_record


# Attention mechanism
# Currently only 'dot' is implemented
# please refer to http://www.aclweb.org/anthology/D15-1166 section 3.1 for more details about Attention implementation
# Input : Decoder state                      with shape [batch size, 1, decoder hidden dimension]
#         Compressed feature from Listner    with shape [batch size, T, listener feature dimension]
# Output: Attention score                    with shape [batch size, T (attention score of each time step)]
#         Context vector                     with shape [batch size,  listener feature dimension]
#         (i.e. weighted (by attention score) sum of all timesteps T's feature)
class Attention(nn.Module):
    def __init__(self, mlp_preprocess_input, preprocess_mlp_dim, use_gpu,activate,mode='dot', input_feature_dim=512):
        super(Attention,self).__init__()
        self.mode = mode.lower()
        self.mlp_preprocess_input = mlp_preprocess_input
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.use_gpu = use_gpu
        self.float_type = torch.torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
        if mlp_preprocess_input:
            self.preprocess_mlp_dim  = preprocess_mlp_dim
            self.phi = nn.Linear(input_feature_dim,preprocess_mlp_dim)
            self.psi = nn.Linear(input_feature_dim,preprocess_mlp_dim)
            self.activate = getattr(F,activate)

    def forward(self, decoder_state, listener_feature):
        if self.mlp_preprocess_input:
            comp_decoder_state = self.relu(self.phi(decoder_state))
            comp_listener_feature = self.relu(TimeDistributed(self.psi, listener_feature))
            # print("****************", np.shape(comp_decoder_state), np.shape(comp_listener_feature), np.shape(listener_feature))
        else:
            comp_decoder_state = decoder_state
            comp_listener_feature = listener_feature
        count_original = 0
        count_copula = 0

        energy = torch.bmm(comp_decoder_state, comp_listener_feature.transpose(1, 2)).squeeze(dim=1)
        #     # print("energy" , np.shape(energy))
        energy_num = energy.data.cpu().numpy()
        if (np.shape(energy_num)[0] <= 3) or Isanycolumnzero(energy_num):
            count_original = 1
            attention_score = self.softmax(energy)
        else:
            CopPDF = np.ones((np.shape(energy_num)[0], np.shape(energy_num)[1]))

            marginal_PDF_E = norm.pdf(energy_num)

            for i in range(0, np.shape(energy_num)[1] - 1):
                #tmp_check = True
                # print("uuu shape" ,np.shape(uuu), "marginalpdf" , np.shape(marginal_PDF_E))
                # print ("uuu ****", uuu)
                if i == 0:
                    CopPDF[:, i] = 1
                else:
                    energy_num2 = np.expand_dims(energy_num[:, i - 1], axis=1)
                    energy_num3 = np.expand_dims(energy_num[:, i], axis=1)

                    # if np.shape(energy_num2) != np.shape(energy_num3):
                    # print("The dimension is wrong")
                    # raise AttributeError("The dimension is wrong"
                    uuu1 = norm.cdf(energy_num2)
                    uuu2 = norm.cdf(energy_num3)
                    uuu = np.concatenate((energy_num2, energy_num3), axis=1)
                    # print("uuu", np.shape(uuu))

                    print("energy_num2" , np.shape(energy_num2), np.shape(energy_num3))
                    #cor_E = np.corrcoef(energy_num2, energy_num3)
                    cor_E = np.corrcoef(energy_num2.transpose(), energy_num3.transpose())
                    # cov_E = np.cov(energy_num2.transpose(), energy_num3.transpose())
                    #print("cor_E", np.shape(cor_E))
                    # if np.shape(cor_E)[0]!=np.shape(cor_E)[1]:
                    # if not is_squared(cor_E):
                    # print("cor_E is and its shape", cor_E, np.shape(cor_E))
                    # raise AttributeError("The matrix must be square matrix")
                    if np.isnan(cor_E).any():
                        # print("energy_num2 and energy_num3", energy_num2, energy_num3)
                        # print("cor_E is and its shape", cor_E, np.shape(cor_E))
                        raise AttributeError("The matrix contais nan!!!!!")
                    else:
                        pass

                    # print("corre shape " , np.linalg.det(cor_E))
                    if np.linalg.det(cor_E) <1e-2 or (np.array_equal(energy_num2,energy_num3)):
                        #CopPDF[:, i] = copulapdf('gaussian', uuu, cor_E)
                        #print("coppdf is huge _copula ", CopPDF[:, i])
                        CopPDF[:, i] = 10
                        #print("marginal pde", marginal_PDF_E)
                        #print("coppdf is huge ",CopPDF[:, i] )
                        # tmp_check = False
                        # print("Determinant is too low", np.linalg.det(cor_E))
                        # #print("equality" ,energy_num2, energy_num3 )
                        # attention_score = self.softmax(energy)
                        # count_original = 1
                        # break

                    # continue
                    else:
                        CopPDF[:, i] = copulapdf('gaussian', uuu, cor_E)
                        #print("tmp_check_checkl",CopPDF[:, i])
                    # print("pde_cop",np.shape(CopPDF[:, i]))
                    # CopPDF[:, i] =np.expand_dims(coptemp, axis =1)
            # print("coppde ", np.shape(CopPDF))
            #print("cop************", CopPDF)
            energy_f1 = np.multiply(marginal_PDF_E, CopPDF)
            energy_f1 = self.float_type(energy_f1)
            #print("energy *************", energy_f1)
            attention_score = self.softmax(energy_f1)
            count_copula = 1
            #print("Copula is happy")

            # print("attention_score 2 ", np.shape(attention_score))
            # attention_score = attention_score.repeat(2,1)
            # print("attention_score 3 ", np.shape(attention_score))
        #print("attension_score *************", attention_score)
        # print("attension_score ##############", np.shape(attention_score.unsqueeze(2).repeat(1,1,listener_feature.size(2))))
        context = torch.sum(listener_feature * attention_score.unsqueeze(2).repeat(1, 1, listener_feature.size(2)),
                            dim=1)
        # print("context",np.shape(context),np.shape(listener_feature))
        return attention_score, context, count_original, count_copula









