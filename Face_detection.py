
""" Face Detection using Multi-Task Cascaded Convolution Neural Network """
	
# Importing the required libraries
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN

 
# definging a function for drawing an image with detected faces and the facial landmarks
def draw_image_with_boxes(imagename, result_list):
	# loading the image
	data = pyplot.imread(imagename)
	# plotting the image
	pyplot.imshow(data)
	# getting the context for drawing boxes
	ax = pyplot.gca()
	# plotting each box
	for result in result_list:
		# get coordinates of the bottom left of the bounding box 
		x, y, width, height = result['box']
		# creating the shape of the boundary box (rectangle)
		rect = Rectangle((x, y), width, height, fill=False, color='red')
		# drawing the box
		ax.add_patch(rect)
		# drawing dots at the facial landmarks of the box
		for key, value in result['keypoints'].items():
			# create and draw dot
			dot = Circle(value, radius=2, color='red')
			ax.add_patch(dot)
	# Saving the plot with the boxes
	pyplot.savefig('After/after12.png', bbox_inches='tight',dpi=100)

    
# loading image 
imagename = 'Before/before12.jpg'
pixels = pyplot.imread(imagename)
# creating the detector, using default weights
detector = MTCNN()
# detecting faces in the image
faces = detector.detect_faces(pixels)

# saving the output of the detector (predicted faces) in a text file
bboxes_file=open("Boundary_boxes.txt","a")
bboxes_file.write('Faces of image '+ imagename +'\n'+'\n')
for face in faces:
     bboxes_file.write(str(face) +'\n')
     
bboxes_file.write('\n'+'\n'+'\n')  
bboxes_file.close()

# displaying the faces on the original image
draw_image_with_boxes(imagename, faces)


