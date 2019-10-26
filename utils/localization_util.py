import numpy as np
from utils.seed_util import retrieve_random_state

def multi_random_localization(canvas_size, images, overlap=False, corner=False, random_seed=False):
    '''Randomly fits multiple images onto a canvas of canvas size.
    
    No overlap? We recommend to have canvas_size >= 6*image_size for 4 images, 
      as otherwise there might not be enough space to place each image.
      
    Parameter overlap determines whether there can be some or no overlap.
    overlap=False: No stamps will overlap.
    overlap=True: All stamps can overlap in any member.
    overlap=(int): Stamps can overlap up to (int) pixels. (not yet implemented)
    '''
    
    r = retrieve_random_state(random_seed)
    
    # Size of the image.
    image_size = images[0].shape[0]
    
    # Canvas for the new image.
    new_image = np.zeros(shape=(canvas_size, canvas_size))
    
    # When overlapping, make sure that the max of image stays at 1.
    max_image = np.ones(shape=(canvas_size, canvas_size))
    
    # Size of places where we can put some points.
    place_size = canvas_size - image_size + 1
    
    # Keep a 2D array of available spots with indices = place
    all_spots = np.reshape(list(range(place_size*place_size)), (place_size, place_size))
    
    # Keep track of a boolean mask for available places.
    if corner:
        available_mask = np.zeros((place_size, place_size), dtype=bool)
        available_mask[0,0] = True
        available_mask[0,-1] = True
        available_mask[-1,0] = True
        available_mask[-1,-1] = True
        overlap = False
    else:
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
        
        coords.append([y_place, x_place])

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