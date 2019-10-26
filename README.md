# StampNet Information

Code for ["StampNet: unsupervised multi-class object discovery"](https://arxiv.org/abs/1902.02693) by Visser, Corbetta, Menkovski and Toschi.

## Installation

### Dependencies

The StampNet code has been written in Python 3 and are dependent on the following packages:

```bash
pip install tensorflow-gpu keras sacred matplotlib scikit-image scikit-learn jupyterlab seaborn munkres opencv-python
```

Dependencies:

- Keras verion: 2.3.1
- Tensorflow: 2.0.0
- Numpy: 1.17.3
- Sacred: 0.8.0

### Datasets

Dataset **CT-MNIST** should be generated before use, which can be done by:

```bash
python generate_cluttered_mnist.py
```

The **Pedestrian** dataset is inside the data folder and needs to be extracted, e.g. 

```bash
tar -zxcf pedestrian.tar.gz
```

## Code explanation

StampNet has been written in Keras using Sacred for experiment production. The parameters for each experiment are stored in their respective python file, e.g. `ct-mnist-2.py`, which calls the main network `stamp-network.py` with the specific parameters. Then:

- The experiment will run
- Experiment data, such as the weights or the parameters, will be stored in the `runs` folder. Each file has their own folder and every run will be stored separately.

The `stampnet_analysis.ipynb` contains a few figures and measures of the paper, which has been tested with Jupyter lab:

```bash
jupyter lab
```

**Experiment 0** contains a sample run for each of the different files (except for T-MNIST-1).
