import pydub
import sys
import os
import re

#set locations for dataset thats downloaded
dataset_location="/Users/pvedul714/Downloads/6901475/" # path to data set folder. 
category1_audio_file_path = dataset_location + "MSoS_challenge_2018_Development_v1-00/Development/Human/"
category2_audio_file_path = dataset_location + "MSoS_challenge_2018_Development_v1-00/Development/Nature/"
category3_audio_file_path = dataset_location + "MSoS_challenge_2018_Development_v1-00/Development/Urban/"


#set output folders where we can write audio files and its lables.
project_folder = "/Users/pvedul714/Documents/AI/project_data_set"
project_training_set_input = project_folder + "/training_set/input_folder/"
project_training_set_output = project_folder + "/training_set/output_folder/"
output_csv_with_audio_file_path= project_training_set_output + "labels.csv"

# open the csv file supplied with the downloaded dataset and process it as dict.
input_csv_with_audio_file_path = dataset_location + "/MSoS_challenge_2018_Development_v1-00/Logsheet_Development.csv"
input_csv_with_audio_files={} # empty dict to store the values
with open(input_csv_with_audio_file_path, "r") as input_csv_file_handle:
  for row in input_csv_file_handle.readlines():
    values=re.split(',',row)
    #values.rstrip()
    input_csv_with_audio_files[values[2].rstrip()]=values[0]+','+values[1]
  #print input_csv_with_audio_files

# check if output  directory exists if not create it. this is where we will write the overlayed outputfiles and its labels.
if   not os.path.isdir(project_folder + "/training_set") :
  os.mkdir(project_folder + "/training_set", 0770)
if not os.path.isdir(project_training_set_input):
  os.mkdir(project_training_set_input, 0770)
if not os.path.isdir(project_training_set_output) :
  os.mkdir(project_training_set_output, 0770)

# process audio files, overlay, write them to output folder.
output_csv_file_handle= open(output_csv_with_audio_file_path, "w+")
output_csv_file_handle.write("Category,Event,File\n")
category1_files=os.listdir(category1_audio_file_path)
category3_files=os.listdir(category3_audio_file_path)
category2_files=os.listdir(category2_audio_file_path)
number_of_files_to_process_per_folder=5

for  category1_audio_file in category1_files[:number_of_files_to_process_per_folder]: 
  output_csv_file_handle.write(input_csv_with_audio_files[category1_audio_file]+","+category1_audio_file+"\n")
  category1_audio_file_handle=pydub.AudioSegment.from_wav(category1_audio_file_path+category1_audio_file)
  category1_audio_file_handle.export(project_training_set_output+category1_audio_file) # write individual audio file to output folder

  for category2_audio_file in category2_files[:number_of_files_to_process_per_folder]: 
        #output_csv_file_handle.write("\n")
    category2_audio_file_handle=pydub.AudioSegment.from_wav(category2_audio_file_path+category2_audio_file)
    category2_audio_file_handle.export(project_training_set_output+category2_audio_file)  # write individual audio file to output folder
    output_csv_file_handle.write(input_csv_with_audio_files[category2_audio_file]+","+category2_audio_file+"\n")
    for category3_audio_file in category3_files[:number_of_files_to_process_per_folder]: 
      category3_audio_file_handle=pydub.AudioSegment.from_wav(category3_audio_file_path+category3_audio_file)
      Overlay_category1_category2_category3_unique_indetifier = category3_audio_file_handle.overlay(category2_audio_file_handle)
      Overlay_category1_category2_category3_unique_indetifier = Overlay_category1_category2_category3_unique_indetifier.overlay(category1_audio_file_handle)
      
      output_audio_file_name=category1_audio_file.split('.')[0]+"_" +category2_audio_file.split('.')[0] + "_"+category3_audio_file.split('.')[0] +".wav"
      Overlay_category1_category2_category3_unique_indetifier.export(project_training_set_input+output_audio_file_name) #save the overlayed audio file
      
      category3_audio_file_handle.export(project_training_set_output+category3_audio_file)  # write individual audio file to output folder
      
      output_csv_file_handle.write(input_csv_with_audio_files[category3_audio_file]+","+category3_audio_file+"\n")
output_csv_file_handle.close()
  #output_csv_file_handle.write(input_csv_with_audio_files[category1_audio_file],category1_audio_file,"\n")

