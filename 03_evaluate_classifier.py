import pickle #for loading your trained classifier

from extract_features import extract_features #our feature extraction

# The function that should classify new images. 
# The image and mask are the same size, and are already loaded using plt.imread
def classify(img):
    """
    Insert only name of image, must be of type ".png"    
    """
        
     #Extract features (the same ones that you used for training)
    x = extract_features(img)
         
     
     #Load the trained classifier
    classifier = pickle.load(open('groupNJ_classifier.sav', 'rb'))
    
    
     #Use it on this example to predict the label AND posterior probability
    pred_label = classifier.predict(x)
    pred_prob = classifier.predict_proba(x)
     
     
     #print('predicted label is ', pred_label)
     #print('predicted probability is ', pred_prob)
    return pred_label, pred_prob
 
    
# The TAs will call the function above in a loop, for external test images/masks