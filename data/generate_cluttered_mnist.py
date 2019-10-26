import numpy as np
from keras.datasets import mnist

from cv2 import resize

batch_size = 64

import numpy as np

def rrs(random_seed):
    """ Returns a random state if the seed is not a random state.
    """
    if not random_seed:
        r = np.random.RandomState(None)
    elif type(random_seed) == np.random.RandomState:
        r = random_seed
    else:
        r = np.random.RandomState(random_seed)
        
    return r

# Retrieving test set
(x_train, y_train), (x_test, y_test) = mnist.load_data()

def mnist_random_local(P, image, random_seed=False):
    ''' Puts MNIST patch randomly in a PxP area, 
    padded with zeros.    
    '''
    
    r = rrs(random_seed)
    
    image_width = image.shape[0]
    new_image = np.zeros(shape=(P,P))
    
    # Grab 2 random ints for X and Y coordinate.
    # Between 0 and P-image_width.
    x_tl, y_tl = r.randint(0, P-image_width+1, 2)
    new_image[y_tl:y_tl+image_width, x_tl:x_tl+image_width] = image

    return new_image, x_tl, y_tl

def multi_mnist_random_local(canvas, images, overlap=False, random_seed=False):
    '''Randomly fits multiple images onto a canvas of canvas size.
    
    No overlap? We recommend to have canvas_size >= 6*image_size for 4 images, 
      as otherwise there might not be enough space to place each image.
      
    Parameter overlap determines whether there can be some or no overlap.
    overlap=False: No stamps will overlap.
    overlap=True: All stamps can overlap in any member.
    overlap=(int): Stamps can overlap up to (int) pixels. (not yet implemented)
    '''
    
    r = rrs(random_seed)
    
    canvas_size = canvas.shape[0]
    
    # Size of the image.
    image_size = images[0].shape[0]
    
    new_image = canvas
    
    # When overlapping, make sure that the max of image stays at 1.
    max_image = np.ones(shape=(canvas_size, canvas_size))
    
    # Size of places where we can put some points.
    place_size = canvas_size - image_size + 1
    
    # Keep a 2D array of available spots with indices = place
    all_spots = np.reshape(list(range(place_size*place_size)), (place_size, place_size))
    
    # Keep track of a boolean mask for available places.
    available_mask = np.ones((place_size, place_size), dtype=bool)
    
    coords = []
    
    for image in images:
        # Check if there are still some available spots left.
        if True not in available_mask:
            raise Exception("No available spots to place the image. " +
                            "Try a higher canvas_size or less images.")
        
        # Retrieve a random index from all available spots.
        place_index = r.choice(all_spots[available_mask].flatten())
        
        # Calculate the (y,x) coordinate from this index.
        y_place = place_index // place_size
        x_place = place_index % place_size
        
        coords.append((y_place, x_place))
        
        # Put the image on this index.
        new_image[y_place:y_place+image_size, x_place:x_place+image_size] += image
        
        new_image = np.minimum(max_image, new_image)
        
        if not overlap:
            # Calculate which places are not allowed anymore. 
            # Should not go out of bounds.
            y_lower = max(0, y_place - image_size + 1)
            x_lower = max(0, x_place - image_size + 1)
            
            # Higher coordinate is calculated 1 higher due to python slicing
            y_higher = min(canvas_size, y_place+image_size)
            x_higher = min(canvas_size, x_place+image_size)

            # Make all potentional overlapping spots False.
            available_mask[y_lower:y_higher, x_lower:x_higher] = False

    # Return the canvas with all images placed.
    return new_image, coords

def generate_cluttered_mnist(x, y, canvas_shape, nr_of_clutters, clutter_shape, random_state=None):
    r = rrs(random_state)
    
    rand_img_idx_list = r.permutation(list(range(x.shape[0])))
    rand_clutter_idx_list = np.swapaxes([r.permutation(list(range(x.shape[0]))) for i in range(nr_of_clutters)], 0, 1)
    images_new = []
    images_labels_new = []
    images_coords = []
    for img_idx, clutter_idx_list in zip(rand_img_idx_list, rand_clutter_idx_list):
        empty_canvas = np.zeros(canvas_shape)
        
        image_to_place = x[img_idx]
        label = y[img_idx]
        
        canvas_image, coords = multi_mnist_random_local(empty_canvas, [image_to_place], overlap=True, random_seed=r.randint(1e9))

        clutter_to_place = [resize(x[i], dsize=clutter_shape) for i in clutter_idx_list]
        
        cluttered_canvas_image, _ = multi_mnist_random_local(canvas_image, clutter_to_place, overlap=True, random_seed=r.randint(1e9))
        
        # Append them on a correct list.
        images_new.append(cluttered_canvas_image)
        images_labels_new.append(label)
        images_coords.append(coords[0])
        
    return np.array(images_new), np.array(images_labels_new), np.array(images_coords)

def generate_cluttered_2_mnist(x, y, canvas_shape, nr_of_clutters, clutter_shape, random_state=None):
    r = rrs(random_state)
    
    rand_img_idx_list = np.swapaxes([r.permutation(list(range(x.shape[0]))) for i in range(2)], 0, 1)
    rand_clutter_idx_list = np.swapaxes([r.permutation(list(range(x.shape[0]))) for i in range(nr_of_clutters)], 0, 1)
    images_new = []
    images_labels_new = []
    images_coords = []
    for img_idx_list, clutter_idx_list in zip(rand_img_idx_list, rand_clutter_idx_list):
        empty_canvas = np.zeros(canvas_shape)
        
        images_to_place = [x[i] for i in img_idx_list]
        labels = [y[i] for i in img_idx_list]
        
        canvas_image, coords = multi_mnist_random_local(empty_canvas, images_to_place, overlap=True, random_seed=r.randint(1e9))

        clutter_to_place = [resize(x[i], dsize=clutter_shape) for i in clutter_idx_list]
        
        cluttered_canvas_image, _ = multi_mnist_random_local(canvas_image, clutter_to_place, overlap=True, random_seed=r.randint(1e9))
        
        # Append them on a correct list.
        images_new.append(cluttered_canvas_image)
        images_labels_new.append(labels)
        images_coords.append(coords)
        
    return np.array(images_new), np.array(images_labels_new), np.array(images_coords)

def generate_and_save_cluttered_mnist(file_name, random_seed):
    r = rrs(random_seed)
    
    print("Generating train data...")
    x_train_new, y_train_new, x_train_coords = generate_cluttered_mnist(x_train, y_train, canvas_shape=(100, 100), nr_of_clutters=8, clutter_shape=(8,8), random_state=r.randint(1e9))
    print("Training data generated! ")
    
    print("Generating test data...")
    x_test_new, y_test_new, x_test_coords = generate_cluttered_mnist(x_test, y_test, canvas_shape=(100, 100), nr_of_clutters=8, clutter_shape=(8,8), random_state=r.randint(1e9))
    print("Test data generated! ")
    
    print("Saving data...")
    np.savez_compressed(
        file_name,
        x_train=x_train_new,
        y_train=y_train_new,
        x_test=x_test_new,
        y_test=y_test_new,
        x_train_coords=x_train_coords,
        x_test_coords=x_test_coords
    )
    print("Data saved!")
    
def generate_and_save_2_cluttered_mnist(file_name, random_seed):
    r = rrs(random_seed)
    
    print("Generating train data...")
    x_train_new, y_train_new, x_train_coords = generate_cluttered_2_mnist(x_train, y_train, canvas_shape=(100, 100), nr_of_clutters=8, clutter_shape=(8,8), random_state=r.randint(1e9))
    print("Training data generated! ")
    
    print("Generating test data...")
    x_test_new, y_test_new, x_test_coords = generate_cluttered_2_mnist(x_test, y_test, canvas_shape=(100, 100), nr_of_clutters=8, clutter_shape=(8,8), random_state=r.randint(1e9))
    print("Test data generated! ")
    
    print("Saving data...")
    np.savez_compressed(
        file_name,
        x_train=x_train_new,
        y_train=y_train_new,
        x_test=x_test_new,
        y_test=y_test_new,
        x_train_coords=x_train_coords,
        x_test_coords=x_test_coords
    )
    print("Data saved!")
    
generate_and_save_cluttered_mnist("cluttered_mnist/8x8-clutter-8-times", 43)
generate_and_save_2_cluttered_mnist("cluttered_mnist/2-shape-8x8-clutter-8-times", 42)