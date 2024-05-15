# Loading the trained classifier
import pickle

from extract_features import extract_features #our feature extraction

# The function that classifies new images. 
# The image and mask must be the same size, and only the filename must be given to the function
def classify(img):
    """
    Insert only filename of image, must be of type ".png" | File Structure must be in accordance with README
    """
        
     # Extract features (the same ones that we used for training)
    x = extract_features(img)
    
    # Load the trained classifier
    classifier = pickle.load(open('groupNJ_classifier.sav', 'rb'))
    
    # Use it on this example to predict the label AND posterior probability
    pred_label = classifier.predict(x)
    pred_prob = classifier.predict_proba(x)
     
    return pred_label, pred_prob
 
    
# The TAs will call the function above in a loop, for external test images/masks
