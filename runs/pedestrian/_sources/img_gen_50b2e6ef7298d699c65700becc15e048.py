import numpy as np
import itertools
import os
import imageio
import json
import math
import utils.rssg as rssg
from utils.seed_util import retrieve_random_state as rrs
from utils.localization_util import multi_random_localization
from sklearn.model_selection import train_test_split
from keras.datasets import mnist



def get_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255

    return (x_train, y_train), (x_test, y_test)


def get_cluttered_mnist(data_path,
                        test_only=False,
                        random_seed=False):
    clutter_data = np.load(data_path)
    x_train = False
    y_train = False
    x_train_coords = False


    if not test_only:
        x_train = clutter_data["x_train"]
        y_train = clutter_data["y_train"]

    x_test = clutter_data["x_test"]
    y_test = clutter_data["y_test"]

    if not test_only:
        x_train_coords = clutter_data["x_train_coords"]
    x_test_coords = clutter_data["x_test_coords"]

    return (x_train, y_train), (x_test, y_test), (x_train_coords, x_test_coords)


def get_ped_data(shuffle=True,
                 shape_size=40,
                 data_path="data/pedestrian_data",
                 random_seed=False):
    """
    Returns the pedestrian dataset.
    """
    r = rrs(random_seed)
    all_images = []
    
    def get_upper_left_coord(coord, shape_size):
        return (int(math.ceil(coord[0])) - shape_size // 2,
                int(math.ceil(coord[1])) - shape_size // 2)

    print("Loading pedestrian data...")
    
    json_path = os.path.join(data_path, "ped_info.json")

    if not os.path.exists(json_path):
        raise ValueError("Could not retrieve data from path: " + str(json_path))
        
    # Load the coordinate data.
    all_coords = []
    with open(json_path, 'r') as input_file:
            coord_dict = json.load(input_file)

    for minute in sorted(os.listdir(data_path)):
        # We only want directories.
        if not minute.isdigit():
            continue

        print(("Minute number: " + str(minute)))

        # Load the filename
        minute_dir = os.path.join(data_path, minute)
        coord_minute_dict = coord_dict[minute]

        for image_name in sorted(os.listdir(minute_dir)):
            # We only want images.
            if image_name[0] == '.':
                continue
            
            upper_left = coord_minute_dict[image_name]
            image_path = os.path.join(minute_dir, image_name)

            try:
                # Read the image and normalize.
                ped_image = imageio.imread(image_path)
                ped_image = ped_image.astype('float32')
                ped_image /= 255
                all_images.append(ped_image)
                
                # Gather coordinate data.
                all_coords.append(get_upper_left_coord(
                    (upper_left[0], upper_left[1]), shape_size))
                
            except Exception as e:
                print("An error has occurred loading an image. Skipping...")
                print(("Exception: " + str(e)))

    all_ped_images = np.array(all_images)
    
    # Split the training and test data in the same manner.
    fixed_seed = r.randint(1e9)

    x_train, x_test = train_test_split(
        all_ped_images, test_size=0.2, shuffle=False, random_state=fixed_seed)

    all_coords = np.array(all_coords)
    x_train_coords, x_test_coords = train_test_split(
        all_coords, test_size=0.2, shuffle=False, random_state=fixed_seed)

    if shuffle:
        combined_train = list(zip(x_train, x_train_coords))
        combined_test = list(zip(x_test, x_test_coords))

        r.shuffle(combined_train)
        r.shuffle(combined_test)

        x_train[:], x_train_coords[:] = list(zip(*combined_train))
        x_test[:], x_test_coords[:] = list(zip(*combined_test))
        
    print("Pedestrian data loaded!")
    
    return (x_train, None), (x_test, None), (None, None)

def get_shapes(shapes_to_use=False,
               shape_size=28,
               train_size=60000,
               test_size=10000,
               random_seed=False):
    """ Returns random shapes as defined in shapes_to_use.
    Defaults to MNIST-like parameters in terms of shape and size.
    """
    
    r = rrs(random_seed)

    if not shapes_to_use:
        raise ValueError(
            "Need to define shapes_to_use: " + str(shapes_to_use))

    canvas_size = shape_size
    nr_of_shapes = len(shapes_to_use)

    x_train = np.empty((train_size, shape_size, shape_size))
    y_train = np.empty((train_size))

    x_test = np.empty([test_size, shape_size, shape_size])
    y_test = np.empty([test_size])
    
    shape_images = []
    
    # Pre-generate the different images
    for shape in shapes_to_use:
        x_image, _, _ = rssg.generate_random_image(
            target_image=np.zeros((canvas_size, canvas_size)),
            shape_size=shape_size,
            flush_out=False,
            shapes_to_use=[shape],
            output_type="upper-left",
            random_seed=r.randint(1e9)
        )
        shape_images.append(x_image)
        
    shape_images = np.array(shape_images)
    
    train_img_idx = r.randint(nr_of_shapes, size=train_size)
    test_img_idx = r.randint(nr_of_shapes, size=test_size)
    
    x_train = shape_images[train_img_idx]
    y_train = train_img_idx
    
    x_test = shape_images[test_img_idx]
    y_test = test_img_idx

    return (x_train, y_train), (x_test, y_test)    


def localize_all_images(images,
                        image_labels,
                        canvas_size=84,
                        nr_img_per_canvas=2,
                        overlap=False,
                        random_seed=False):
    """ 
    Place the images on an empty canvas in order to create the dataset.
    Parameter 'overlap' indicates whether these images should overlap or not.
    """
    r = rrs(random_seed)

    images_new = []
    images_labels_new = []
    images_coords = []

    # Permutate the images for placing them on a canvas.
    rand_img_idx_list = np.swapaxes([r.permutation(
        list(range(images.shape[0]))) for i in range(nr_img_per_canvas)], 0, 1)

    # We put nr_img_per_canvas images randomly on the canvas.
    for rand_img_idx in rand_img_idx_list:
        # Retrieve the images we need to place according to the current rand_img_idx
        images_to_place = [images[i] for i in rand_img_idx]

        # Retrieve their respective labels
        labels = [image_labels[i] for i in rand_img_idx]

        # Retrieve a localized version on a single canvas.
        canvas_image, coords = multi_random_localization(canvas_size=canvas_size,
                                                         images=images_to_place,
                                                         overlap=overlap,
                                                         random_seed=r.randint(1e9))
        # Append them on a correct list.
        images_new.append(canvas_image)
        images_labels_new.append(labels)
        images_coords.append(coords)

    images_new = np.array(images_new)
    images_labels_new = np.array(images_labels_new)
    images_coords = np.array(images_coords)

    return np.array(images_new), np.array(images_labels_new), np.array(images_coords)


def get_localized_images(canvas_size=84,
                         nr_img_per_canvas=2,
                         img_type='shape_loc',
                         overlap=False,
                         random_seed=False,
                         test_only=False,
                         **kwargs):
    """ 
    Returns a dataset of images containing the shapes randomly placed on a canvas.
    """
    r = rrs(random_seed)


    print("Loading image data...")
    if 'shape' in img_type:
        shapes_to_use = kwargs.pop('shapes_to_use', False)
        (x_train, y_train), (x_test, y_test) = get_shapes(
            random_seed=r.randint(1e9), shapes_to_use=shapes_to_use)
    elif 'mnist' in img_type:
        (x_train, y_train), (x_test, y_test) = get_mnist()
    else:
        raise Exception("Localized image type '", img_type,
                        "' is not currently supported.")

    print("Loaded!")
    
    if test_only:
        x_train_loc = y_train_loc = train_coords = False
    else:
        print("Localizing train data...")
        x_train_loc, y_train_loc, train_coords = localize_all_images(
            images=x_train,
            image_labels=y_train,
            canvas_size=canvas_size,
            nr_img_per_canvas=nr_img_per_canvas,
            overlap=overlap,
            random_seed=r.randint(1e9))
        print("Training data localized!")
    
    print("Localizing test data...")

    x_test_loc, y_test_loc, test_coords = localize_all_images(
        images=x_test,
        image_labels=y_test,
        canvas_size=canvas_size,
        nr_img_per_canvas=nr_img_per_canvas,
        overlap=overlap,
        random_seed=r.randint(1e9))
    
    print("Test data localized!")

    return (x_train_loc, y_train_loc), (x_test_loc, y_test_loc), (train_coords, test_coords)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def get_batch_and_tensors(data_list,
                          output_shapes,
                          cycle=False,
                          shuffle=False,
                          random_seed=None):
    """ Returns a generator that generates batches of the data_list.
    The output_shapes list is used to generate an extra output
      of zeros, in order to match the additional outputs of the model.
    """

    r = rrs(random_seed)

    batch_size = output_shapes[0][0]

    data_chunks = chunks(data_list, batch_size)

    if cycle:
        data_iterator = itertools.cycle(data_chunks)
    else:
        data_iterator = data_chunks

    def shuffler(x): return r.random.shuffle(x)

    for chunked_data in data_iterator:
        dt = np.stack(chunked_data)

        # Shuffle if required
        if shuffle:
            dt = shuffler(dt)

        # Gather the outputs by putting the input as the first output.
        #  and zeros for all other outputs (correctly sized).
        final_outputs = [dt]

        for i in range(len(output_shapes) - 1):
            final_outputs.append(
                np.zeros(output_shapes[i+1])
            )

        yield (dt, final_outputs)

def get_img_data(img_rows, 
                 img_cols, 
                 img_type, 
                 img_type_parameters,
                 test_only=False, 
                 random_seed=False):
                 
    r = rrs(random_seed)

    x_train = False
    x_test = False

    if img_type == "t_shape":
        (x_train, _), (x_test, _), (_, _) = get_localized_images(
            canvas_size=img_rows,
            img_type=img_type,
            shapes_to_use=img_type_parameters["shapes_to_use"],
            shape_size=img_type_parameters["shape_size"],
            nr_img_per_canvas=img_type_parameters["nr_img_per_canvas"],
            overlap=img_type_parameters["overlap"],
            test_only=test_only,
            random_seed=r.randint(1e9))
    
    elif img_type == 't_mnist':
        (x_train, _), (x_test, _), (_, _) = get_localized_images(
            canvas_size=img_rows,
            img_type=img_type,
            nr_img_per_canvas=img_type_parameters["nr_img_per_canvas"],
            overlap=img_type_parameters['overlap'],
            test_only=test_only,
            random_seed=r.randint(1e9))

    elif img_type == "ct_mnist":
        (x_train, _), (x_test, _), (_, _) = get_cluttered_mnist(
            img_type_parameters["data_path"], 
            test_only=test_only,
            random_seed=r.randint(1e9))

    elif img_type == 'pedestrian':
        (x_train, _), (x_test, _), (_, _) = get_ped_data(
            data_path=img_type_parameters["data_path"],
            shuffle=img_type_parameters["shuffle"],
            random_seed=r.randint(1e9))
    else:
        raise ValueError("Image type " + str(img_type) +
                            " is not currently supported.")

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    return x_train, x_test

def retrieve_data(
        output_shapes,
        img_type,
        img_type_parameters=False,
        steps_per_epoch=128,
        validation_steps=128,
        shuffle=False,
        test_only=False,
        random_seed=None,
        **kwargs):

    r = rrs(random_seed)

    # Retrieve information about the input / output image.
    batch_size = output_shapes[0][0]
    img_rows, img_cols = output_shapes[0][1], output_shapes[0][2]


    # Return the training and test data depending on the image type.
    # No labels will be returned as the network is an autoencoder.
    x_train, x_test = get_img_data(img_rows,
                                   img_cols,
                                   img_type,
                                   img_type_parameters,
                                   test_only=test_only,
                                   random_seed=r)

    # Shuffle again to make sure that the dataset is shuffled.
    if shuffle:
        r.shuffle(x_train)
        r.shuffle(x_test)
    
    # Training image generator.
    def gen_img():
        for x in x_train:
            yield x

    # Test image generator.
    def gen_img_test():
        for x in x_test:
            yield x

    gen = itertools.cycle(gen_img())
    gen_test = itertools.cycle(gen_img_test())

    def get_new_image():
        im = next(gen)
        return im

    def get_new_test_image():
        im = next(gen_test)
        return im

    training_size = steps_per_epoch * batch_size
    training_data = [get_new_image() for _ in range(training_size)]

    val_size = validation_steps * batch_size
    val_data = [get_new_image() for _ in range(val_size)]

    test_size = steps_per_epoch * batch_size
    test_data = [get_new_test_image() for _ in range(test_size)]

    training_generator = itertools.cycle(
        get_batch_and_tensors(training_data,
                              output_shapes=output_shapes,
                              shuffle=shuffle,
                              random_seed=r.randint(1e9))
    )

    val_generator = itertools.cycle(
        get_batch_and_tensors(val_data,
                              output_shapes=output_shapes,
                              shuffle=shuffle,
                              random_seed=r.randint(1e9))
    )

    test_generator = itertools.cycle(
        get_batch_and_tensors(test_data,
                              output_shapes=output_shapes,
                              shuffle=shuffle,
                              random_seed=r.randint(1e9))
    )

    return {
        "training_generator": training_generator,
        "val_generator": val_generator,
        "test_generator": test_generator
    }
