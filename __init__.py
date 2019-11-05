from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import torchaudio
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import config
import io_processing

class AutoEncoderModel(nn.Module):
	def __init__(self):
		super(AutoEncoderModel, self).__init__()

		self.encode_cnn = nn.Sequential(
			nn.ZeroPad2d((config.padding_horiz_size[0], config.padding_horiz_size[1], 0, 0)),
			nn.Conv2d(config.num_inp_channels, config.num_conv_horiz_channels, config.conv_horiz_filter_size),
			nn.ZeroPad2d((0, 0, config.padding_vert_size[0], config.padding_vert_size[1])),
			nn.Conv2d(config.num_conv_horiz_channels, config.num_conv_vert_channels, config.conv_vert_filter_size)
		)
		
		self.shared_fcl = nn.Sequential(
			nn.Linear(config.num_fcc_in_features, config.num_fcc_out_features),
			nn.ReLU()
		)

		self.fcl_speech = nn.Sequential(
			nn.Linear(config.num_fcc_out_features, config.num_fcc_in_features),
			nn.ReLU()
		)

		self.deconv_speech = nn.Sequential(
			nn.ZeroPad2d((0, 0, config.padding_vert_size[0], config.padding_vert_size[1])),
			nn.Conv2d(config.num_conv_vert_channels, config.num_conv_horiz_channels, config.conv_vert_filter_size),
			nn.ZeroPad2d((config.padding_horiz_size[0], config.padding_horiz_size[1], 0, 0)),
			nn.Conv2d(config.num_conv_horiz_channels, config.num_inp_channels, config.conv_horiz_filter_size)
		)

		self.fcl_baby_cry = nn.Sequential(
			nn.Linear(config.num_fcc_out_features, config.num_fcc_in_features),
			nn.ReLU()
		)

		self.deconv_baby_cry = nn.Sequential(
			nn.ZeroPad2d((0, 0, config.padding_vert_size[0], config.padding_vert_size[1])),
			nn.Conv2d(config.num_conv_vert_channels, config.num_conv_horiz_channels, config.conv_vert_filter_size),
			nn.ZeroPad2d((config.padding_horiz_size[0], config.padding_horiz_size[1], 0, 0)),
			nn.Conv2d(config.num_conv_horiz_channels, config.num_inp_channels, config.conv_horiz_filter_size)
		)

		self.fcl_siren = nn.Sequential(
			nn.Linear(config.num_fcc_out_features, config.num_fcc_in_features),
			nn.ReLU()
		)

		self.deconv_siren = nn.Sequential(
			nn.ZeroPad2d((0, 0, config.padding_vert_size[0], config.padding_vert_size[1])),
			nn.Conv2d(config.num_conv_vert_channels, config.num_conv_horiz_channels, config.conv_vert_filter_size),
			nn.ZeroPad2d((config.padding_horiz_size[0], config.padding_horiz_size[1], 0, 0)),
			nn.Conv2d(config.num_conv_horiz_channels, config.num_inp_channels, config.conv_horiz_filter_size)
		)

		self.fcl_dog = nn.Sequential(
			nn.Linear(config.num_fcc_out_features, config.num_fcc_in_features),
			nn.ReLU()
		)

		self.deconv_dog = nn.Sequential(
			nn.ZeroPad2d((0, 0, config.padding_vert_size[0], config.padding_vert_size[1])),
			nn.Conv2d(config.num_conv_vert_channels, config.num_conv_horiz_channels, config.conv_vert_filter_size),
			nn.ZeroPad2d((config.padding_horiz_size[0], config.padding_horiz_size[1], 0, 0)),
			nn.Conv2d(config.num_conv_horiz_channels, config.num_inp_channels, config.conv_horiz_filter_size)
		)

		self.out_ReLU = nn.ReLU();

	def forward(self, inp):
		encode_out = self.encode_cnn(inp)
		shared_fc_out = self.shared_fcl(encode_out)

		decode_speech = self.fcl_speech(shared_fc_out)
		decode_speech = self.deconv_speech(decode_speech)

		decode_baby_cry = self.fcl_speech(shared_fc_out)
		decode_baby_cry = self.deconv_baby_cry(decode_baby_cry)

		decode_siren = self.fcl_speech(shared_fc_out)
		decode_siren = self.deconv_siren(decode_siren)

		decode_dog = self.fcl_speech(shared_fc_out)
		decode_dog = self.deconv_dog(decode_dog)

		concat_out = torch.cat((decode_speech, decode_baby_cry, decode_siren, decode_dog), 1)
		concat_out = torch.cat((decode_baby_cry, decode_siren, decode_dog), 1)
		concat_out = self.out_ReLU(concat_out)

		return concat_out

def train(model):
	model.train()
	
	loss_list = []

	for epoch in range(config.num_epochs):
		for batch_idx, (data, target) in enumerate(io_processing.train_loader):
			model_out = model(data[0].permute(0, 3, 1, 2))
			
			for i in range(len(target)):
				target[i] = target[i][:, 0, :, :].permute(0, 3, 1, 2)
			
			concat_target = torch.cat((target[0], target[1], target[2]), 1)
			
			loss = criterion(model_out, concat_target)
			loss_list.append(loss.item())

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if batch_idx % 10 == 0:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					epoch, batch_idx * len(data), len(io_processing.train_loader.dataset),
					100. * batch_idx / len(io_processing.train_loader), loss
				))

model = AutoEncoderModel()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
criterion = nn.MSELoss()

train(model)