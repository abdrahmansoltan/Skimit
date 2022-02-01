import numpy as np
import json
import tensorflow as tf
import tensorflow_text as text
import spacy
from spacy.lang.en import English

# load_option = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
skimlit_model = tf.keras.models.load_model('./data/model/')
# skimlit_model = tf.keras.models.load_model('Models/Skimlit_BertModel', options=load_option)

def spacy_function(abstract):
    
    # setup English sentence parser
    nlp = English()

    # create sentence splitting pipeline object
    sentencizer = nlp.create_pipe("sentencizer")

    # add sentence splitting pipeline object to sentence parser
    nlp.add_pipe('sentencizer')
    
    # create "doc" of parsed sequences, change index for a different abstract
    doc = nlp(abstract) 

    # return detected sentences from doc in string type (not spaCy token type)
    abstract_lines = [str(sent) for sent in list(doc.sents)]
    
    return abstract_lines
    
# ---------------------------------------------------------------------------------------------------------------------------

def split_chars(text):
    return ' '.join(list(text))

# ---------------------------------------------------------------------------------------------------------------------------

def make_predictions(text):
    
    classes = ['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS']
    abstract_lines = list()
    
    abstract_lines = spacy_function(text)
    
    # Get total number of lines
    total_lines_in_sample = len(abstract_lines)

    # Go through each line in abstract and create a list of dictionaries containing features for each line
    sample_lines = []
    for i, line in enumerate(abstract_lines):
        sample_dict = {}
        sample_dict["text"] = str(line)
        sample_dict["line_number"] = i
        sample_dict["total_lines"] = total_lines_in_sample - 1
        sample_lines.append(sample_dict)
    
    # Get all line_number values from sample abstract
    test_abstract_line_numbers = [line["line_number"] for line in sample_lines]

    # One-hot encode to same depth as training data, so model accepts right input shape
    test_abstract_line_numbers_one_hot = tf.one_hot(test_abstract_line_numbers, depth=15)
    
    # Get all total_lines values from sample abstract
    test_abstract_total_lines = [line["total_lines"] for line in sample_lines]
    
    # One-hot encode to same depth as training data, so model accepts right input shape
    test_abstract_total_lines_one_hot = tf.one_hot(test_abstract_total_lines, depth=20)
    
    # Split abstract lines into characters
    abstract_chars = [split_chars(sentence) for sentence in abstract_lines]
    
    # Make predictions on sample abstract features
    test_abstract_pred_probs = skimlit_model.predict(x=(test_abstract_line_numbers_one_hot,
                                                       test_abstract_total_lines_one_hot,
                                                       tf.constant(abstract_lines),
                                                       tf.constant(abstract_chars)))
    
    test_abstract_preds = tf.argmax(test_abstract_pred_probs, axis=1)
    
    test_abstract_pred_classes = [classes[i] for i in test_abstract_preds]

    return (test_abstract_pred_classes, abstract_lines)