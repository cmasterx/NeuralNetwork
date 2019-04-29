NOTE: This program was initially submitted on time. The difference between this submission and the previous
is that this submission contains the pillow package necessary to parse the bdf font. I initially had trouble getting pip
to work properly in my local machine and was initially unable to have the package within the submission file.

The assignment code, "FinalProject.py" contains the whole program for the assignment. The only change in this file from the
previous submission are lines 32 and 33 which are:

32	import sys
33	sys.path.insert(0, './packages')

to allow the program to use the missing package. Without these lines, the program will thrown an error for the missing packages.
All other lines of code besides lines 32 and 33 are exactly the same between the two submissions. Besides "FinalProject.py and this
README file, no additional files have been changed between the first submission and this submission.
The Final Report (FinalReport.pdf) is the same exact file between this and the previous submissions.


========== HOW TO RUN ==========
The main program is called ‘FinalProject.py’ and it runs on Python 3.6. 

The first 30 lines of the python file contains Boolean flags to test various components of the algorithm, 
including training the algorithm and testing the robustness of the program by adding random noise.

- The TRAIN flag is initially set to False, but when True, the program will begin training a new neural network. This will automatically generate a neural network
is called img_recog_ai. This will output the final accuracy of the neural network after 1500 epoch.
- The GRAPHING flag is initally set to False. When true, this will use matplotlib to graph, when applicable, parts of the program such as in Training or Testing Noise
Accuracy.
	*** Compute does not have matplotlib package, so setting this flag to True will throw an error. I'm unable to include this in the submission due to large file
		size of the package

- NW_FILENAME: is the file name to load the previously trained neural network from saved data. This neural network will be loaded if TRAIN is set to False

- TEST_NOISE_ACCURACY: Initially set to False, generates a graph of neural network accuracy as noise increases
	*** Compute does not have matplotlib package, so setting this flag to True will throw an error.

- TEST_NOISE_LETTER Initially set to True, this test each letter by adding noise until it fails
	This will output the number of noises added to each letter to cause the neural network to fail recognition for that character

- TEST_MAJORITY Initially set to False, this tests the majority function with the neural network where it is true if both inputs are 1



========== HOW TO USE NEURAL NETWORK CLASS ==========
Example on how to create neural network object:
img_recog_ai = NeuralNetwork(arr)
img_recog_ai = NeuralNetwork(arr, alpha)
	- arr is an integer array of at least two elements
	- the first element in arr is the number of inputs
	- the last element in arr is the number of outputs
	- each element in between the first and last elements are the hidden layers,
	  and the value of each element is the number of neurons
		* For example NeuralNetwork([2, 10, 2]) has 2 inputs, 2 outputs, and 1 hidden layer with 10 neurons
	- alpha is the learning rate and is optional

img_recog_ai = NeuralNetwork.load('File_name.npy')
	- loads neural network from saved data

img_recog_ai.save('File_name.npy')
	- saves neural network to file

img_recog_ai.output(input_matrix)
	- input_matrix: an n by 1 matrix where n is the number of inputs in the neural network and each cell
		is a single input
	- returns an m by 1 matrix where m is the number of outputs in the neural network and each cell
		is a single output

ex = Example(input, res)
	- a class that contains an input and expected result of the algorithm
	- ex.input returns the input, an n by 1 matrix where n is the number of inputs in the neural network and each cell
		is a single input
	- ex.res returns the expected output, an m by 1 matrix where m is the number of outputs in the neural network and each cell
		is a single output

img_recog_ai.learn(example_list)
	- uses back propagation to teach the neural network
	- this is a list, so teaching with one example will be example_list = [ex] where ex is an Example class

gcl = generate_char_list()
	- generates a list of Letters and information about the image. This is important for the following methods not listed in this README file


~ The remainding methods in the "FinalProject.py" are document within the file. If the function contains a parameter call "char_list", gcl will be used.
  For example: generate_example(gcl).
~ For methods with parameter "char", this is a value between 0 and 25, representing A - Z. If "char" is -1, then a randomized letter is chosen.

