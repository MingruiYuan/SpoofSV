import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class textEmbedding(nn.Module):

	def __init__(self, vocab_len, out_channels=128):
		"""
		Text Embedding Layer.
		Args:
		--vocab_len, types of characters. {type = int}
		--out_channels. {type = int}
		"""
		super(textEmbedding, self).__init__()
		self.vocab_len = vocab_len

		# Each column is a embedding.
		# self.W = torch.zeros((out_channels, vocab_len), requires_grad=True)
		# nn.init.kaiming_normal_(self.W, nonlinearity='relu')
		self.W = nn.Linear(in_features=vocab_len, out_features=out_channels)

	def forward(self, inputs):
		"""
		--inputs:(Batch, 1, Chars). They have been converted to textid(int).
		"""
		inputs = inputs.long()

		# One-hot encoding. (Batch*Vocab_len*Chars)
		one_hot = torch.zeros((inputs.shape[0], self.vocab_len, inputs.shape[2])).to(device).scatter_(1, inputs, torch.ones(inputs.shape).to(device))
		#print(one_hot.device)
		outputs = self.W(one_hot.permute(0,2,1)) #(Batch*Chars*out_channels)
		return outputs.permute(0,2,1)

class highwayConv(nn.Module):
	"""
	This is a highway convolution layer which is used in some models.
	It will use "same" convolution.
	"""

	def __init__(self, dimension, kernel_size, dilation, causal=False):
		"""
		Args:
		--dimension. Number of input and output channels. They always keep same in highway nets.
		--causal. A boolean on whether to use causal convolution or ordinary one.
		"""
		super(highwayConv, self).__init__()
		self.dimension = dimension
		self.causal = causal

		# Same Convolution. Stride = 1.
		# L_out = L_in + 2*padding - dilation*(kernel_size - 1)
		# As in this model, kernel_size is always odd,
		# we do not consider the case that pad is not an integer.
		self.pad = dilation*(kernel_size-1) // 2

		self.conv = nn.Conv1d(in_channels=dimension, out_channels=2*dimension, kernel_size=kernel_size, padding=0 if causal else self.pad, dilation=dilation)
		self.ln1 = nn.LayerNorm(normalized_shape=dimension)
		self.ln2 = nn.LayerNorm(normalized_shape=dimension)

	def forward(self, inputs):
		"""
		--inputs: (Batch, dimension, timeseries)
		--x: (Batch, 2*dimension, timeseries)
		--H1/H2: (Batch, dimension, timeseries)
		--outputs: (Batch, dimension, timeseries)
		"""

		# zero-padding prior to the inputs to ensure causal convolution.
		if self.causal and self.pad>0:
			d1, d2, d3 = inputs.size()
			inputs = torch.cat((torch.zeros((d1, d2, 2*self.pad)).to(device), inputs), dim=-1)

		#print('hc inputs device: ', inputs.device)

		x = self.conv(inputs)
		H1 = x[:, :self.dimension, :]
		H2 = x[:, self.dimension:, :]
		H1 = self.ln1(H1.permute(0,2,1)).permute(0,2,1)
		H2 = self.ln2(H2.permute(0,2,1)).permute(0,2,1)
		outputs = F.sigmoid(H1)*H2+(1-F.sigmoid(H1))*inputs[:, :, 2*self.pad if self.causal else 0:]
		return outputs

class highwayDilationIncrement(nn.Module):

	def __init__(self, dimension, causal=False):
		"""
		--causal. A boolean on whether to use causal convolution or ordinary one.
		"""
		super(highwayDilationIncrement, self).__init__()

		self.hc1 = highwayConv(dimension=dimension, kernel_size=3, dilation=1, causal=causal)
		self.hc2 = highwayConv(dimension=dimension, kernel_size=3, dilation=3, causal=causal)
		self.hc3 = highwayConv(dimension=dimension, kernel_size=3, dilation=9, causal=causal)
		self.hc4 = highwayConv(dimension=dimension, kernel_size=3, dilation=27, causal=causal)

	def forward(self, inputs):
		x = self.hc1(inputs)
		x = self.hc2(x)
		x = self.hc3(x)
		x = self.hc4(x)
		return x

class textEncoder(nn.Module):

	def __init__(self, vocab_len, textemb_dim=128, hidden_dim=256):
		super(textEncoder, self).__init__()
		self.hidden_dim = hidden_dim

		# Text Embedding Layer
		self.textemb_layer = textEmbedding(vocab_len=vocab_len, out_channels=textemb_dim)

		self.conv1 = nn.Conv1d(in_channels=textemb_dim, out_channels=2*hidden_dim, kernel_size=1)
		self.ln1 = nn.LayerNorm(normalized_shape=2*hidden_dim)
		self.conv2 = nn.Conv1d(in_channels=2*hidden_dim, out_channels=2*hidden_dim, kernel_size=1)
		self.ln2 = nn.LayerNorm(normalized_shape=2*hidden_dim)
		self.hci1 = highwayDilationIncrement(dimension=2*hidden_dim)
		self.hci2 = highwayDilationIncrement(dimension=2*hidden_dim)
		self.hc1 = highwayConv(dimension=2*hidden_dim, kernel_size=3, dilation=1)
		self.hc2 = highwayConv(dimension=2*hidden_dim, kernel_size=3, dilation=1)
		self.hc3 = highwayConv(dimension=2*hidden_dim, kernel_size=1, dilation=1)
		self.hc4 = highwayConv(dimension=2*hidden_dim, kernel_size=1, dilation=1)

	def forward(self, inputs):
		x = self.textemb_layer(inputs)
		x = self.conv1(x)
		x = self.ln1(x.permute(0,2,1)).permute(0,2,1)
		x = self.conv2(F.relu(x))
		x = self.ln2(x.permute(0,2,1)).permute(0,2,1)
		x = self.hci1(x)
		x = self.hci2(x)
		x = self.hc1(x)
		x = self.hc2(x)
		x = self.hc3(x)
		x = self.hc4(x)
		K = x[:, :self.hidden_dim, :]
		V = x[:, self.hidden_dim:, :]
		return K, V

class audioEncoder(nn.Module):

	def __init__(self, freq_bins, hidden_dim=256, condition=False, spkemb_dim=None):
		super(audioEncoder, self).__init__()

		# Two fully-connected layers which project speaker embeddings into the model.
		self.condition = condition
		if condition:
			self.fc1 = nn.Linear(in_features=spkemb_dim, out_features=hidden_dim)
			self.fc2 = nn.Linear(in_features=spkemb_dim, out_features=hidden_dim)

		# kernel_size=1, then whether causal or not is trivial.
		self.conv1 = nn.Conv1d(in_channels=freq_bins, out_channels=hidden_dim, kernel_size=1)
		self.ln1 = nn.LayerNorm(normalized_shape=hidden_dim)
		self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)
		self.ln2 = nn.LayerNorm(normalized_shape=hidden_dim)
		self.conv3 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)
		self.ln3 = nn.LayerNorm(normalized_shape=hidden_dim)

		self.hci1 = highwayDilationIncrement(dimension=hidden_dim, causal=True)
		self.hci2 = highwayDilationIncrement(dimension=hidden_dim, causal=True)
		self.hc1 = highwayConv(dimension=hidden_dim, kernel_size=3, dilation=3, causal=True)
		self.hc2 = highwayConv(dimension=hidden_dim, kernel_size=3, dilation=3, causal=True)

	def forward(self, inputs, spk=None):
		"""
		Args:
		--inputs: Ground truth mel-spec for current timeseries. (Batch*Freq_bins*Current_timeseries)
		--spk: Speaker embeddings. (Batch*Spkemb_dim*1, broadcasting along time)
		"""
		if self.condition:
			x = self.conv1(inputs)
			s = self.fc1(spk.permute(0,2,1)).permute(0,2,1)
			x = self.ln1((x+s).permute(0,2,1)).permute(0,2,1)
			x = self.conv2(F.relu(x))
			x = self.ln2(x.permute(0,2,1)).permute(0,2,1)
			x = self.conv3(F.relu(x))
			p = self.fc2(spk.permute(0,2,1)).permute(0,2,1)
			x = self.ln3((x+p).permute(0,2,1)).permute(0,2,1)
			x = self.hci1(x)
			x = self.hci2(x)
			x = self.hc1(x)
			x = self.hc2(x)
		else:
			x = self.conv1(inputs)
			x = self.ln1(x.permute(0,2,1)).permute(0,2,1)
			x = self.conv2(F.relu(x))
			x = self.ln2(x.permute(0,2,1)).permute(0,2,1)
			x = self.conv3(F.relu(x))
			x = self.ln3(x.permute(0,2,1)).permute(0,2,1)
			x = self.hci1(x)
			x = self.hci2(x)
			x = self.hc1(x)
			x = self.hc2(x)
		return x

class audioDecoder(nn.Module):

	def __init__(self, freq_bins, hidden_dim=256):
		super(audioDecoder, self).__init__()

		self.conv1 = nn.Conv1d(in_channels=2*hidden_dim, out_channels=hidden_dim, kernel_size=1)
		self.ln1 = nn.LayerNorm(normalized_shape=hidden_dim)
		self.hci = highwayDilationIncrement(dimension=hidden_dim, causal=True)
		self.hc1 = highwayConv(dimension=hidden_dim, kernel_size=3, dilation=1, causal=True)
		self.hc2 = highwayConv(dimension=hidden_dim, kernel_size=3, dilation=1, causal=True)
		self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)
		self.ln2 = nn.LayerNorm(normalized_shape=hidden_dim)
		self.conv3 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)
		self.ln3 = nn.LayerNorm(normalized_shape=hidden_dim)
		self.conv4 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)
		self.ln4 = nn.LayerNorm(normalized_shape=hidden_dim)
		self.conv5 = nn.Conv1d(in_channels=hidden_dim, out_channels=freq_bins, kernel_size=1)
		self.ln5 = nn.LayerNorm(normalized_shape=freq_bins)

	def forward(self, inputs):
		x = self.conv1(inputs)
		x = self.ln1(x.permute(0,2,1)).permute(0,2,1)
		x = self.hci(x)
		x = self.hc1(x)
		x = self.hc2(x)
		x = self.conv2(x)
		x = self.ln2(x.permute(0,2,1)).permute(0,2,1)
		x = self.conv3(F.relu(x))
		x = self.ln3(x.permute(0,2,1)).permute(0,2,1)
		x = self.conv4(F.relu(x))
		x = self.ln4(x.permute(0,2,1)).permute(0,2,1)
		x = self.conv5(F.relu(x))
		x = self.ln5(x.permute(0,2,1)).permute(0,2,1)
		x = F.sigmoid(x)
		return x

class melSyn(nn.Module):

	def __init__(self, vocab_len, condition, spkemb_dim, textemb_dim=128, freq_bins=80, hidden_dim=256):
		"""
		Model: From text to mel-spectrogram.
		Args:
		--vocab_len: Types of characters in the vocabulary.
		--textemb_dim: Dimension of text embeddings.
		--freq_bins: Number of mel filter banks.
		--hidden_dim: Common dimension of channels within the model.
		--step: string. 'train' or 'synthesize'.
		"""
		super(melSyn, self).__init__()
		self.hidden_dim = hidden_dim
		#self.attention = Attention()
		self.text_encoder = textEncoder(vocab_len=vocab_len, textemb_dim=textemb_dim, hidden_dim=hidden_dim)
		self.audio_encoder = audioEncoder(freq_bins=freq_bins, hidden_dim=hidden_dim, condition=condition, spkemb_dim=spkemb_dim)
		self.audio_decoder = audioDecoder(freq_bins=freq_bins, hidden_dim=hidden_dim)

	def forward(self, melspec, textid, spkemb, K=None, V=None, A_last=None, pma=None):
		"""
		Args:
		--textid: Inputs of text encoder. Already coverted to numbers. (Batch*1*Chars)
		--melspec: Inputs of audio encoder. (Batch*freq_bins*Total_timeseries)
		"""
		T = melspec.shape[-1] # Total timeseries.
		B = melspec.shape[0] # Batch size.
		#N = textid.shape[-1]

		if self.training:
			K, V = self.text_encoder(textid) # (Batch*hidden_dim*Chars)
			Q = self.audio_encoder(melspec, spkemb)  # (Batch*hidden_dim*current_Timeseries)
			A = torch.matmul(K.permute(0,2,1), Q) / math.sqrt(self.hidden_dim)
			A = F.softmax(A, dim=1)
			# max_attention = torch.argmax(self.A, dim=1)
			R = torch.matmul(V, A)
			R = torch.cat((R, Q), dim=1)
			Y_prob = self.audio_decoder(R)

			return Y_prob, A			
		
		else:
			if T == 1:
				K, V = self.text_encoder(textid)
			N = K.shape[-1]

			Q = self.audio_encoder(melspec, spkemb)
			A = torch.matmul(K.permute(0,2,1), Q) / math.sqrt(self.hidden_dim)
			for k in range(B):
				if pma[k] > 0:
					A[k, :pma[k], -1] = -2**32
				if pma[k] + 2 < N-1:
					A[k, pma[k]+3:, -1] = -2**32

			A = F.softmax(A, dim=1)
			if T > 1:
				A = torch.cat((A_last, A[:, :, -1:]), dim=-1)
			max_attention = torch.argmax(A, dim=1)

			R = torch.matmul(V, A)
			R = torch.cat((R, Q), dim=1)
			Y_prob = self.audio_decoder(R)

			if T == 1:
				return Y_prob, A, max_attention[:, -1], K, V
			else:
				return Y_prob, A, max_attention[:, -1]


class upsampling(nn.Module):
	"""
	1d deconvlution + 2*highway convolution.
	"""
	def __init__(self, ssrn_dim):
		super(upsampling, self).__init__()
		self.deconv = nn.ConvTranspose1d(in_channels=ssrn_dim, out_channels=ssrn_dim, kernel_size=2, stride=2)
		self.hc1 = highwayConv(dimension=ssrn_dim, kernel_size=3, dilation=1)
		self.hc2 = highwayConv(dimension=ssrn_dim, kernel_size=3, dilation=3)

	def forward(self, inputs):
		x = self.deconv(inputs)
		x = self.hc1(x)
		x = self.hc2(x)
		return x

class SSRN(nn.Module):

	def __init__(self, freq_bins, output_bins, ssrn_dim):
		super(SSRN, self).__init__()
		self.conv1 = nn.Conv1d(in_channels=freq_bins, out_channels=ssrn_dim, kernel_size=1)
		self.ln1 = nn.LayerNorm(normalized_shape=ssrn_dim)
		self.hc1 = highwayConv(dimension=ssrn_dim, kernel_size=3, dilation=1)
		self.hc2 = highwayConv(dimension=ssrn_dim, kernel_size=3, dilation=3)
		self.ups1 = upsampling(ssrn_dim=ssrn_dim)
		self.ups2 = upsampling(ssrn_dim=ssrn_dim)
		self.conv2 = nn.Conv1d(in_channels=ssrn_dim, out_channels=2*ssrn_dim, kernel_size=1)
		self.ln2 = nn.LayerNorm(normalized_shape=2*ssrn_dim)
		self.hc3 = highwayConv(dimension=2*ssrn_dim, kernel_size=3, dilation=1)
		self.hc4 = highwayConv(dimension=2*ssrn_dim, kernel_size=3, dilation=1)
		self.conv3 = nn.Conv1d(in_channels=2*ssrn_dim, out_channels=output_bins, kernel_size=1)
		self.ln3 = nn.LayerNorm(normalized_shape=output_bins)
		self.conv4 = nn.Conv1d(in_channels=output_bins, out_channels=output_bins, kernel_size=1)
		self.ln4 = nn.LayerNorm(normalized_shape=output_bins)
		self.conv5 = nn.Conv1d(in_channels=output_bins, out_channels=output_bins, kernel_size=1)
		self.ln5 = nn.LayerNorm(normalized_shape=output_bins)
		self.conv6 = nn.Conv1d(in_channels=output_bins, out_channels=output_bins, kernel_size=1)
		self.ln6 = nn.LayerNorm(normalized_shape=output_bins)

	def forward(self, inputs):
		x = self.conv1(inputs)
		x = self.ln1(x.permute(0,2,1)).permute(0,2,1)
		x = self.hc1(x)
		x = self.hc2(x)
		x = self.ups1(x)
		x = self.ups2(x)
		x = self.conv2(x)
		x = self.ln2(x.permute(0,2,1)).permute(0,2,1)
		x = self.hc3(x)
		x = self.hc4(x)
		x = self.conv3(x)
		x = self.ln3(x.permute(0,2,1)).permute(0,2,1)
		x = self.conv4(x)
		x = self.ln4(x.permute(0,2,1)).permute(0,2,1)
		x = self.conv5(F.relu(x))
		x = self.ln5(x.permute(0,2,1)).permute(0,2,1)
		x = self.conv6(F.relu(x))
		x = self.ln6(x.permute(0,2,1)).permute(0,2,1)
		x = F.sigmoid(x)
		return x