
# Galaxy identification
In this project, you will be examining and classifying images of galaxies based on their shape, a task that poses a significant challenge even for humans. The project has been adapted from a Kaggle challenge, but the tasks you will undertake are simpler than those in the original challenge.

The dataset for this project was generated using crowdsourcing, with non-experts asked to categorize images following a specific taxonomy. Each image was assigned a floating-point number between zero and one, based on the fraction of participants who assigned the image to a given class. This means that the targets of this dataset are not binary labels, but floating-point numbers.

The full project is a regression problem, but it has been broken down into a simpler classification problem and a regression problem. The tasks assigned do not encompass the full complexity of the original challenge. If you wish to tackle the original challenge and submit it as your project, you are welcome to do so. However, note that the corresponding Kaggle challenge is now outdated, and the techniques that won this challenge may not be as interesting to consider. For image classification, there are other, more recent challenges on Kaggle that may be more engaging.

As with all projects, you can start out using a notebook, but the final product should be in Python form. The project is structured into three subtasks listed below.

## Dataset
This dataset contains about 60,000 RGB images of galaxies together with a label vector of 37 dimensions. These labels correspond to answers given to 11 Questions. Each label is a number between 0 and 1 and shows the amount of participants who gave this answer relative to the number of participants that saw the given image.

In order to run this notebook, you have to download and extract the dataset (images: images_training_rev1.zip, labels: training_solutions_rev1.zip) from the Kaggle competition page. Note that the image files have 424x424 pixels, while the galaxies are contained within the central 207x207 pixels. It is useful to crop the central part and reduce to 64x64 pixels.
### Download data
You can download the image datasets from [here](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data) (You have to be logged in).
The file you should download `images_train_rev1.zip` into `data/images` and unzip it (`unzip images_train_rev1.zip`). The labels related to the train file can be found in `data/labels.csv`.

### FlowChart
Flowchart visualizing the question can be seen [here](https://storage.googleapis.com/kaggle-media/competitions/kaggle/3175/media/Screen%20Shot%202013-09-25%20at%2010.08.17.png)

### Additional requirements
You might have to install `torchvision` and `PIL`:
```
pip install torchvision
```
and 
```
pip install pillow
```

## Aims
This project is divided into several subtasks.

### Exercise 0: Data Exploration
Initially, familiarize yourself with the galaxy data and create some diagnostic plots.
<!-- #### Data exploration -->
- Identify any potential issues with the data.
- Discuss possible problems with the data recording process.

If you implement any, remember to report them!

### Exercise 1: Classification
Once you are comfortable with the data, proceed to create a classification model, which aim to determine if a galaxy is **(1) simply smooth and round with no disk, (2) has a disk, or (3) if the image is flawed.**

In this exercise use columns: `[Class1.1, Class1.2, Class1.3]` of `labels.csv`

- Convert the probabilities into one-hot encoded labels and apply filtering as needed.
- Develop a classification model.
- Report key metrics of performance.
- Provide information for result reproduction.

### Exercise 2: Regression

In the second exercise, perform a regression task using columns: `[Class2.1, Class2.2]` and `[Class7.1, Class7.2, Class7.3]` of `labels.csv`.

`[Class2.1, Class2.2]` answers the question:
- Could this be a disk viewed edge-on?

while `[Class7.1, Class7.2, Class7.3]` answers the question:
- How round is the smooth galaxy?

Remember, we do not use one-hot encoded labels, but the original floats that range between 0 and 1.

*Bonus:*
These regression values has a constraint to them. **What is it and could you use it?**
<!-- You should try to make sure that the output of the classifier matches the hierarchical structure of the questions, e.g. the the summed values for Q2 equal the value for answer Q1.1. -->

### Exercise 3: Regression continued

For the third exercise, we further include columns `[Class6.1, Class6.2]` and `[Class8.1, Class8.2, Class8.3, Class8.4, Class8.5, Class8.6, Class8.7]` to the output. This answers additionally the questions
- Is there anything "odd" about the galaxy?
- What is the odd feature?
regarding oddities.

You will have to improve your architecture in order to correctly classify rare object classes.

### Studies
* Does the same architecture (with a different output layer) perform well for all three tasks?
* Can you use augmentations to improve the classification performance? (Especially at test time).
* Can you use the output of a model from one task to inform the prediction of the next
