#!/usr/bin/env python
# coding: utf-8

# 1. LIBRARY IMPORTS
# ==========================================
import numpy as np           # NumPy: Handles mathematical operations and array processing
import pandas as pd          # Pandas: Used for structured data manipulation and DataFrames
import cv2                   # OpenCV: Handles computer vision tasks like reading images and drawing boxes
import pytesseract           # PyTesseract: Wrapper for Google's Tesseract OCR to extract text from images
from glob import glob        # Glob: Unix style pathname pattern expansion (finding files)
import spacy                 # spaCy: Advanced Natural Language Processing (NLP) library
import re                    # Regular Expressions: Used for text cleaning and pattern matching
import string                # String: Provides constants like standard whitespace and punctuation
# import warnings              # Warnings: Provides control over warning messages
# warnings.filterwarnings('ignore') # Suppresses messy warning logs in the console

def clean_text(txt: str) -> str:
    """
    Executes lexical sanitization on the provided string by neutralizing
    extraneous whitespace and non-essential punctuation characters.

    This function normalizes the input to lowercase and sequentially applies
    character translation tables to excise specified morphological artifacts,
    rendering a continuous alphanumeric sequence optimal for downstream
    Natural Language Processing (NLP) or Named Entity Recognition (NER) pipelines.

    Parameters:
    -----------
    txt : str
        The raw input string necessitating lexical normalization.

    Returns:
    --------
    str
        The sanitized, lowercase string devoid of defined whitespaces and punctuation.
    """
    # Delineate the corpus of whitespace characters targeted for excision
    whitespaces = string.whitespace
    
    # Delineate the set of punctuation marks subject to removal. 
    # Note: Essential delimiter characters (e.g., period, hyphen, at-symbol) are intentionally preserved.
    punctuation = '!"#$%&\'()*+:;<=>?[\\]^`{|}~'
    
    # Construct translation matrices mapping the targeted characters to a null value (None)
    table_whitespace = str.maketrans('', '', whitespaces)
    table_punctuation = str.maketrans('', '', punctuation)
    
    # Coerce the input to a string data type to ensure methodological stability
    text = str(txt)
    
    # Apply case normalization to convert all alphabetic characters to lowercase
    #text = text.lower()
    
    # Execute the primary translation to eradicate whitespace artifacts
    text_no_whitespaces = text.translate(table_whitespace)
    
    # Execute the secondary translation to eliminate the specified punctuation artifacts
    text_no_spchar_whitespaces = text_no_whitespaces.translate(table_punctuation)
    
    # Return the finalized, sanitized string representation
    return str(text_no_spchar_whitespaces)

class GroupGen():
    """
    A stateful enumerator designed to assign sequential, unique integer identifiers 
    to contiguous blocks of identical categorical labels.

    This class operates as a temporal tracking mechanism during sequential data 
    iteration. It evaluates state persistence between consecutive observations, 
    amalgamating identical, adjacent entities (e.g., consecutive 'ORG' tokens) 
    into a singular logical group for subsequent spatial aggregation.
    """

    def __init__(self) -> None:
        """
        Initializes the internal state variables governing the sequential 
        identifier and the temporal tracking of the categorical state.
        """
        # The integer enumerator tracking the current active group identifier
        self.id = 0
        
        # The temporal state variable storing the preceding categorical label 
        # to evaluate sequence continuity
        self.text = ' '

    def get_group(self, text: str) -> int:
        """
        Evaluates the continuity of the categorical sequence and assigns an 
        appropriate group identifier.

        Parameters:
        -----------
        text : str
            The current categorical label or token under evaluation.

        Returns:
        --------
        int
            The assigned logical group identifier.
        """
        # Condition 1: State Persistence
        # If the current label matches the preceding state, maintain the existing 
        # enumerator value to cluster these entities together.
        if self.text == text:
            return self.id
            
        # Condition 2: Categorical Divergence
        # If a novel label is encountered, indicating a semantic boundary, increment 
        # the enumerator and update the temporal state.
        else:
            self.id += 1
            self.text = text
            return self.id

def parser(text: str, label: str) -> str:
    # Safety check: Handle missing values (NaNs) or empty strings gracefully
    if not isinstance(text, str) or text.strip() == '':
        return ''
    
    if label == 'PHONE':
        text = text.lower()
        # \D removes everything that is NOT a digit (keeps numbers only)
        text = re.sub(r'\D', '', text)

    elif label == 'EMAIL':
        text = text.lower()
        # CORRECTION: The hyphen (-) is placed at the end to prevent invalid range errors
        # Removes everything except letters, numbers, spaces, or @_.-
        text = re.sub(r'[^a-z0-9@_.\- ]', '', text)

    elif label == 'WEB':
        text = text.lower()
        # CORRECTION: The hyphen (-) is escaped and placed at the end of the character set
        # Removes everything except letters, numbers, spaces, or :/.%#-
        text = re.sub(r'[^a-z0-9:/.%#\- ]', '', text)

    elif label in ('NAME', 'DES'):
        text = text.lower()
        # Removes everything except lowercase alphabetic letters and spaces
        text = re.sub(r'[^a-z ]', '', text)
        # Capitalizes the first letter of each word (e.g., "john doe" -> "John Doe")
        text = text.title()

    elif label == 'ORG':
        text = text.lower()
        # Removes special characters, keeps alphanumeric characters and spaces
        text = re.sub(r'[^a-z0-9 ]', '', text)
        text = text.title()

    # Final cleanup: Collapses multiple spaces down to a single space 
    # (e.g., "John   Doe" -> "John Doe") and strips outer whitespace
    return " ".join(text.split())

def get_predictions(image: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Executes a unified Optical Character Recognition (OCR) and Named Entity Recognition (NER) 
    pipeline to extract, classify, and spatialize semantic entities from an input image.

    The methodology encompasses sequential stages of text acquisition via PyTesseract, 
    lexical sanitization, predictive modeling utilizing a custom spaCy NER architecture, 
    spatial-textual alignment bridging OCR bounding boxes with NLP tokens, spatial 
    aggregation for multi-word entities, and ontological structuring.

    Parameters:
    -----------
    image : np.ndarray
        The input matrix representation of the image (typically loaded via OpenCV).

    Returns:
    --------
    tuple[np.ndarray, dict]
        - img_group_bounding_boxes : A copy of the input image array superimposed 
          with synthesized bounding boxes and ontological labels.
        - entities : A structured dictionary categorizing the extracted strings into 
          predefined semantic classes (e.g., NAME, ORG, PHONE).
    """

    # =========================================================================
    # Phase 1: Optical Character Recognition (OCR) Acquisition
    # =========================================================================
    # Execute PyTesseract to extract textual data and corresponding spatial coordinates
    tess_data = pytesseract.image_to_data(image)
    
    # Parse the Tab-Separated Values (TSV) string into a two-dimensional matrix structure
    tess_list = list(map(lambda x: x.split('\t'), tess_data.split('\n')))
    
    # Instantiate a Pandas DataFrame, utilizing the initial row as the schema header
    df = pd.DataFrame(tess_list[1:], columns=tess_list[0])
    
    # Purge anomalous or null observations to ensure structural integrity
    df.dropna(inplace=True) 

    # =========================================================================
    # Phase 2: Lexical Normalization & Corpus Synthesis
    # =========================================================================
    # Apply lexical sanitization to neutralize morphological artifacts
    df['text'] = df['text'].apply(clean_text)

    # Isolate valid alphanumeric sequences by excluding empty strings
    df_clean_data = df.query('text != "" ')
    
    # Synthesize a continuous corpus string. The heuristic assumes a single 
    # whitespace delineator between sequential words.
    content = " ".join([w for w in df_clean_data['text']])
    print(content)
    
    # =========================================================================
    # Phase 3: Named Entity Recognition (NER) Inference
    # =========================================================================
    # Execute the pre-trained NLP architecture to derive ontological predictions
    doc = model_NER(content)

    # Serialize the resulting spaCy document object into a JSON-compliant schema
    doc_json = doc.to_json()
    doc_text = doc_json['text']

    # Extract foundational tokens derived from the spaCy tokenization engine
    df_tag = pd.DataFrame(doc_json['tokens'])
    
    # Map the absolute character indices back to the synthesized corpus to retrieve token strings
    df_tag['token'] = df_tag[['start', 'end']].apply(
        lambda x: doc_text[int(x['start']):int(x['end'])], axis=1
    )

    # =========================================================================
    # Phase 4: Spatial-Textual Alignment
    # =========================================================================
    # Isolate the predicted semantic labels ('ents') and their starting coordinate bounds
    entities_df = pd.DataFrame(doc_json['ents'])[['start', 'label']]
    
    # Execute a left join to superimpose predicted labels onto the token DataFrame
    tokens_labels_df = pd.merge(df_tag, entities_df, how='left')
    
    # Impute missing values with 'O' (Outside), adhering to the standard BIO annotation schema
    tokens_labels_df.fillna('O', inplace=True)

    # Reconstruct character-level coordinate indices within the OCR DataFrame.
    # The cumulative sum (cumsum) heuristic simulates the spatial length of the concatenated string.
    df_clean_data['end'] = df_clean_data['text'].apply(lambda x: len(x) + 1).cumsum() - 1 
    df_clean_data['start'] = df_clean_data[['text', 'end']].apply(
        lambda x: x['end'] - len(x['text']), axis=1
    )

    # Execute an inner join via the 'start' integer index to fuse spatial OCR 
    # parameters (bounding boxes) with the NLP predictions
    complete_info_df = pd.merge(
        df_clean_data, 
        tokens_labels_df[['start', 'token', 'label']], 
        how='inner', 
        on='start'
    )

    # =========================================================================
    # Phase 5: Spatial Aggregation & Morphological Tagging
    # =========================================================================
    # Exclude non-entity observations to isolate targets requiring bounding boxes
    bounding_boxes_df = complete_info_df.query("label != 'O' ").copy()

    # Excise the positional prefix (B-/I-) from the BIO schema to isolate the core entity class
    bounding_boxes_df['label'] = bounding_boxes_df['label'].apply(lambda x: x[2:])
    
    # Assign sequential identifier groups to amalgamate multi-word entities
    bounding_boxes_df['group'] = bounding_boxes_df['label'].apply(group_gen.get_group)

    # Enforce integer typing for spatial coordinates
    bounding_boxes_df[['left', 'top', 'width', 'height']] = bounding_boxes_df[['left', 'top', 'width', 'height']].astype(int)
    
    # Compute the terminal spatial boundaries (right and bottom coordinates)
    bounding_boxes_df['right'] = bounding_boxes_df['left'] + bounding_boxes_df['width']
    bounding_boxes_df['bottom'] = bounding_boxes_df['top'] + bounding_boxes_df['height']

    # Aggregate spatial parameters by group to generate global bounding boxes for contiguous entities
    col_group = ['left', 'top', 'right', 'bottom', 'label', 'token', 'group']
    group_tag_img = bounding_boxes_df[col_group].groupby(by='group')
    img_tagging = group_tag_img.agg({
        'left': min,
        'right': max,
        'top': min,
        'bottom': max,
        'label': 'first',
        'token': lambda x: " ".join(x)
    })

    # =========================================================================
    # Phase 6: Graphical Annotation
    # =========================================================================
    # Clone the initial image matrix to prevent destructive modification of the original array
    img_group_bounding_boxes = image.copy()
    
    # Iterate through the aggregated spatial matrices to render annotations
    for l, r, t, b, label, token in img_tagging.values:
        # Render the spatial delimiter (bounding box)
        cv2.rectangle(img_group_bounding_boxes, (l, t), (r, b), (0, 255, 0), 2)
        # Render the ontological classification textual overlay
        cv2.putText(img_group_bounding_boxes, label, (l, t), cv2.FONT_HERSHEY_PLAIN, 1.5, (225, 127, 255), 2)

    # =========================================================================
    # Phase 7: Entity Parsing & Ontological Structuring
    # =========================================================================
    # Isolate tokens and their corresponding BIO-tagged labels
    token_label_info = complete_info_df[['token', 'label']].values
    
    # Initialize the structured dictionary to house parsed entities
    entities = dict(NAME=[], ORG=[], DES=[], PHONE=[], EMAIL=[], WEB=[])
    previous = 'O'
    
    # Traverse the array sequentially to concatenate tokens adhering to the BIO logic
    for token, label in token_label_info:
        bio_tag = label[0]        # Positional identifier (Beginning/Inside)
        label_tag = label[2:]     # Semantic class identifier
    
        # Execute secondary parsing constraints (e.g., regex filtering via the parser function)
        text = parser(token, label_tag)
    
        if bio_tag in ('B', 'I'):
            # Detect transitions between distinct ontological classifications
            if previous != label_tag:
                entities[label_tag].append(text)
            else:
                # Detect the initiation of a novel entity within the same classification ('B')
                if bio_tag == 'B':
                    entities[label_tag].append(text)
                # Concatenate the token to the pre-existing entity ('I')
                else:
                    # Specific semantic categories necessitate whitespace separators
                    if label_tag in ('NAME', 'ORG', 'DES'):
                        entities[label_tag][-1] = entities[label_tag][-1] + " " + text
                    # Data structures such as phone numbers or emails require unbroken strings
                    else:
                        entities[label_tag][-1] = entities[label_tag][-1] + text
    
        # Update the temporal state variable for the subsequent iteration
        previous = label_tag
        
    return img_group_bounding_boxes, entities

# ==========================================
# 2. MODEL LOADING
# ==========================================
# Load the custom-trained Named Entity Recognition (NER) model from the specified directory
model_NER = spacy.load('./output/model-best/')

# Instantiate the stateful automaton globally to preserve operational state 
# across iterative DataFrame applications (e.g., during .apply() mapping).
group_gen = GroupGen()


