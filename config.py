#set output folders where we can write audio files and its lables.
project_folder = "./dataset/"
project_dataset_input = project_folder + "input_folder/"
project_dataset_output = project_folder + "output_folder/"
output_csv_with_audio_file_path = project_dataset_output + "labels.csv"

#dataset parameters
train_set_percent = 0.92
dev_set_percent = 1 - train_set_percent

#data processing parameters
hann_window_length = 1024
stft_size = 1024
stft_freq_bins = 513

#model parameters
num_classes = 3
num_inp_channels = 2
num_conv_horiz_channels = 50
num_conv_vert_channels = 30
conv_horiz_filter_size = (1, stft_freq_bins)
conv_vert_filter_size = (15, 1)
padding_horiz_size = (int((conv_horiz_filter_size[1] - 1)/2), conv_horiz_filter_size[1] - 1 - int((conv_horiz_filter_size[1] - 1)/2))
padding_vert_size = (int((conv_vert_filter_size[0] - 1)/2), conv_vert_filter_size[0] - 1 - int((conv_vert_filter_size[0] - 1)/2))
num_fcc_in_features = 188
num_fcc_out_features = 188

#training parameters
num_epochs = 5
batch_size = 1
learning_rate = 0.1