Overview of files and its functions

--- saliency_map_generator.py ---
Creates saliency maps, using an image and a given model, to
check where the neural network looks at in an image. 

--- geometric_gradient_analysis.py ---
This applies the geometric gradient from the paper with the use
of saliency maps to check the trustworthyness of a neural network. 

--- noise_creator.py ---
This function takes an image and applies a simple gaussian noise to it.
We use this to create out of data images to check if works. 

