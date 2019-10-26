import stamp_network
import os
from sacred.observers import FileStorageObserver
from sacred import Experiment

cur_file_name = os.path.basename(__file__)[:-3]

steps_per_epoch = 128
img_rows = 80
stamp_size = 40

config_file = {
    "run_parameters" : {
        "epochs": 1000,
        "steps_per_epoch": 128,
        "validation_steps": 100,
        "shuffle": False,
        # These parameters depend on the image type.
        "img_type": "pedestrian",
        "img_type_parameters": {
            "data_path": "data/pedestrian", 
            "resize": False, 
            "shuffle": True
        }
    },

    "layer_parameters" : {
        "input_parameters": {
            "batch_size": 64,
            "img_rows": img_rows,
            "img_cols": img_rows,
            "img_channels": 1,
        },
        "encoder_parameters": {
            "encoder_dropout_rate": 0.2,
            "conv_layer_sizes": [16, 32, 64, 128, 128],
            "conv_kernel_size": (3, 3),
            "final_conv_size": 256,
            "final_conv_kernel": (3, 3)
        },
        "decoder_parameters": {
            "decoder_enc_size": 128,
            "stamp_size": stamp_size,
            "nr_of_stamps": 40,
            "stamps_per_canvas": 3,
            "gumbel_parameters": {
                "tau_init": 7,
                "anneal_rate": 0.01,
                "min_temperature": 0.2,
                "steps_per_epoch": steps_per_epoch
            },
            "coord_tensor_size": img_rows - stamp_size + 1
        },
        "loss": "mse"
    }
}

stamp_network.ex.add_config(config_file)
stamp_network.ex.observers[0] = FileStorageObserver.create(
        basedir=os.path.join('runs', cur_file_name))
stamp_network.ex.run(options={'--name': cur_file_name})