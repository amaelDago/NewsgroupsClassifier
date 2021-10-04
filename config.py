import os

target_name_dict = {
    0 :  'alt.atheism',
    1 :  'comp.graphics',
    2 :  'comp.os.ms-windows.misc',
    3 :  'comp.sys.ibm.pc.hardware',
    4 :  'comp.sys.mac.hardware',
    5 :  'comp.windows.x',
    6 :  'misc.forsale',
    7 :  'rec.autos',
    8 :  'rec.motorcycles',
    9 :  'rec.sport.baseball',
    10 : 'rec.sport.hockey',
    11 : 'sci.crypt',
    12 : 'sci.electronics',
    13 : 'sci.med',
    14 : 'sci.space',
    15 : 'soc.religion.christian',
    16 : 'talk.politics.guns',
    17 : 'talk.politics.mideast',
    18 : 'talk.politics.misc',
    19 : 'talk.religion.misc'
}
# SET ROOT
ROOT = os.path.abspath(os.path.dirname(__file__))


# SET HYPERPARAMTERS
BATCH_SIZE = 32

# SET MODEL
MODEL_NAME = "sentence-transformers/distilbert-base-nli-mean-tokens"
MODEL_FOLDER = os.path.join(ROOT,"models")
EMBEDDING_SIZE = 768


# TRAIN MEBEDDING
MATRIX_EMBEDDING_PATH = os.path.join(ROOT, "train_embedding/matrix_emdedding.pkl")
LABELS_PATH = os.path.join(ROOT, "train_embedding/train_labels.pkl")

