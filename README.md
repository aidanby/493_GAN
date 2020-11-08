# Requirements
To install requirements
```
pip install -r requirements.txt
```
Use Python version 3.7.4. 

# Training
## Training RenyiGAN-[&beta;<sub>1</sub>, &beta;<sub>2</sub>]
To train RenyiGAN-[&beta;<sub>1</sub>, &beta;<sub>2</sub>], run this command
```
python3 renyigan_varying_alpha.py
```
which will prompt you to input the seed, trial number, version, and subversion. 

The version and subversion determines the values of &beta;<sub>1</sub> and &beta;<sub>2</sub>
in RenyiGAN-[&beta;<sub>1</sub>, &beta;<sub>2</sub>] 
as well as whether or not you activate L1 normalization and/or gradient penalty. 
The table below details this information:

| Version | Subversion | [&beta;<sub>1</sub>, &beta;<sub>2</sub>] | Gradient penalty  | L1 normalization |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 1 | [0, 0.9] | &#10006; | &#10006; |
| 1 | 2 | [0, 3.0] | &#10006; | &#10006; |
| 1 | 3 | [1.1, 4.0] | &#10006; | &#10006; |
| 2 | 1 | [0, 0.9] | &#10006; | &#10004; |
| 2 | 2 | [0, 3.0] | &#10006; | &#10004; |
| 2 | 3 | [1.1, 4.0] | &#10006; | &#10004; |
| 3 | 1 | [0, 0.9] | &#10004; | &#10006; |
| 3 | 2 | [0, 3.0] | &#10004; | &#10006; |
| 3 | 3 | [1.1, 4.0] | &#10004; | &#10006; |
| 4 | 1 | [0, 0.9] | &#10004; | &#10004; |
| 4 | 2 | [0, 3.0] | &#10004; | &#10004; |
| 4 | 3 | [1.1, 4.0] | &#10004; | &#10004; |


The code will automatically download the MNIST database using the `utils.py` file. 
It will also save the generated images as `.npy` files in `data/annealing/`. 
Note that the code only saves the checkpoints for the first trial in the `checkpoints` directory. 

## Training RenyiGAN-&alpha;
To train RenyiGAN-&alpha; (i.e. RenyiGAN for a fixed value &alpha; for all epochs during training), run this command:
```
python3 renyigan_static_alpha.py
``` 
which will prompt you to input the alpha value, trial number, version, and seed. 
We used seed 123, 5005, 1600, 199621, 60677, 20435, 15859, 33764, 79878, 
36123 for trials 1 to 10, respectively.
The version number determines whether or not you activate L1 normalization and/or gradient penalty.
The table below details this information:

| Version | Gradient Penalty | L1 normalization |
| :---: | :---: | :---: |
| 1 | &#10006; | &#10006; |
| 2 | &#10006; | &#10004; |
| 3 | &#10004; | &#10006; |
| 4 | &#10004; | &#10004; |

Note that if you input an &alpha; = 1, then the code will run DCGAN (see paper for proof).

# Evaluation code
The repository has `fid_varying_renyigan.py` and `fid_static_renyigan.py` codes,
which are evaluation codes use to compute the raw FID scores for RenyiGAN-[&beta;<sub>1</sub>, &beta;<sub>2</sub>] 
and RenyiGAN-&alpha; respectively.
To evaluate RenyiGAN-[&beta;<sub>1</sub>, &beta;<sub>2</sub>], simply run
```
python3 fid_varying_renyigan.py
``` 
which will prompt you to enter the version, subversion, and the trial number. 
To evaluate RenyiGAN-&alpha;, simply run
```
python3 fid_static_renyigan.py
``` 
which will prompt you to enter the alpha value and version.
Note that the evaluation scripts use on the numpy (`.npy`) files that the training files save, 
and delete all but the best performing epoch's generated images. 
We sampled 10,000 real and generated images to calculate the FID scores. 

# Further information 
We initialized the weights of each layer by using a Gaussian random variable with mean 0 and standard deviation
0.01.
We also used the Adam optimizer with a learning rate of 
 2 x 10<sup>-4</sup>, &beta;<sub>1</sub> = 0.5, &beta;<sub>2</sub> = 0.999, and
&epsilon; = 1 x 10<sup>-7</sup> for both networks. 
The batch size was chosen to be 100 for the 60,000 MNIST images.
The total number of epochs was 250 for the MNIST images. 

You can find sample images and plots for each version in `images/`. 
