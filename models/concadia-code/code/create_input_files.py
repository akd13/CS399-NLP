from utils import create_input_files
import argparse
import sys

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# Note: We're doing away with the caption/description dichotomy entirely!

#	parser.add_argument('label', type=str, choices=['description', 'caption'],
 #       help='whether the model is optimized to generate descriptions or captions')
	parser.add_argument('context', type=str,
		choices=['context', 'none'],
        help='the type of text that the model receives as additional context to inform generation')
	parser.add_argument('--dataset', type=str, choices=['pew', 'statista', 'concadia', 'hci'], required = True,
        help='what should be randomized for a baseline condition')
	parser.add_argument('--root_dir', type=str, 
		required=True, help='where the raw dataset is stored')
	parser.add_argument('--image_subdir', type=str,
		default="imgs",
		help='the name of the image subdir within `root_dir` that houses a decompressed version of `resized.zip`')
	args = parser.parse_args()

	json_name = args.dataset
#	if args.randomized == "context":
#		json_name = 'wiki_split_randomcontext'
#	elif args.randomized == "img":
#		json_name = 'wiki_split_randomfilename'
#	elif args.randomized == "none":
#		json_name = 'wiki_split'
#	else:
#		json_name = 'random'
#		sys.exit("invalid parser argument -- try img, context, or none.")

	# Create input files (along with word map)

	print("Context ", args.context)

	create_input_files(root_dir=args.root_dir,
		json_path=json_name+'.json',
		image_folder=args.image_subdir,
    	labels_per_image=1,
	    context=args.context,
		# arxiv paper version:
    	# min_word_freq=1,
		min_word_freq=1,
		output_folder='parsed_data/' + args.context + args.dataset + '/',
    	max_len=50)
