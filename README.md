# How to Install and Run the Project

## Required folder structure to run files:
│
├── fyp2024            <- Project repository With all the files (Your are here)  
│
├── data               <- Folder with all the images and masks you want to test
│   ├── images         <- Images Folder
│   ├── masks          <- Masks Folder
│   └── metadata.csv   <- Labels, demographic variables etc
│

Note: Images and mask should be of type png


## How To Run:
### Create model:
Run "01_process_images.py" to extract the features from all of your images.
The features then gets extracted into a csv file on your machine.
Then run "02_train_classifiers.py" with the csv file
Then save the model by running "03_evaluate_classifier.py"

### Test model

