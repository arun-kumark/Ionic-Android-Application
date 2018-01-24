from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import argparse
import numpy as np
import tensorflow as tf

#Global list for Negative Cases
unknown = ['agata potato', 'cashew', 'honneydew melon', 'nectarine', 'spanish pear', 'asterix potato', \
          'fuji apple', 'kiwi', 'onion', 'plum', 'taiti lime', 'diamond peach', 'granny smith apple', \
          'orange', 'watermelon', 'broccoli']
rolls = ['rolls round', 'rolls square', 'rolls bag']
chicken_wings = ['chicken wings uncut','chicken wings cut']
chicken_legs = ['chicken legs uncut', 'chicken legs cut']
french_fries = ['french fries thick', 'french fries thin', 'french fries wavy']
chicken_nuggets = ['chicken nugget cut', 'chicken nugget uncut']

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)
  return graph

def test_model(file_name, label_file):
	print(file_name)
	
	graph = load_graph(model_file)
	t = read_tensor_from_image_file(file_name, 224, 224, 128, 128)

	input_name = "import/" + input_layer
	output_name = "import/" + output_layer
	input_operation = graph.get_operation_by_name(input_name);
	output_operation = graph.get_operation_by_name(output_name);

	with tf.Session(graph=graph) as sess:
		results = sess.run(output_operation.outputs[0],
			{input_operation.outputs[0]: t})
	results = np.squeeze(results)

	top_k = results.argsort()[-5:][::-1]
	labels = load_labels(label_file)

	for i in top_k:
		if labels[i] in unknown:
			print ("Unknown")
		elif labels[i] in chicken_nuggets:
			print("chicken_nuggets")			
		elif labels[i] in rolls:
			print("rolls")
		elif labels[i] in chicken_wings:
			print("chicken_wings")
		elif labels[i] in chicken_legs:
			print("chicken_legs")
		elif labels[i] in french_fries:
			print("french_fries")
		else:
			print(labels[i])
	return

#Some universal settings

input_height = 224
input_width = 224
input_mean = 128
input_std = 128
input_layer = "input"
output_layer = "final_result"

if __name__ == "__main__":
	model_file = os.getcwd()+"/tf_files/retrained_graph.pb"
	label_file = os.getcwd()+"/tf_files/retrained_labels.txt"

	#Chnage directory
	os.chdir("tf_files/food-items")

	#Get inside the food item and grab the list of food items
	list_of_food_items=os.listdir()
	glob_list=[]
	for item in list_of_food_items:
		os.chdir(item)
		for file in os.listdir():
			test_model(str(os.path.abspath(file)),label_file)
		os.chdir('..')





