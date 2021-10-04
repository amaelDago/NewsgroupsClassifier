""""
TO DO : 
0 : Get ONNX MODELS FOR Tokenizer
1 VEctorize train set
2 Index train set
3 Build Knn 
4 
"""
# We choose Faiss because Faiss is a C++ based library built by Facebook
# AI with a complete wrapper in python, to index vectorized data and to perform efficient searches on them
# Many type of index with Faiss exist but according to our data and the guideline of index choose
# (https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)
# choose, we keep  IVF K (where k = 4*sqrt(N) to 16*sqrt(N))


import faiss
import numpy as np
import logging
import config

# Logging Configuration 
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

class SearchEngine(object) :

    def __init__(
        self,
        matrix : np.array,
        labels : list,
        nlist : int = 100,
        embedding_size : int = 768,
    ) : 
        self.matrix = matrix
        self.labels = labels
        self.embedding_size = embedding_size
        self.nlist = nlist

        self.index = self.get_index()

        logging.info("Train...")
        self.index.train(self.matrix)

        self.index.add(self.matrix)

    def get_index(self) : 
        logging.info("Index matrix ...")
        quantizer =  faiss.IndexFlatIP(self.embedding_size)
        index = faiss.IndexIVFFlat(
            quantizer, 
            self.embedding_size,
            self.nlist
        )
        return index

    def predict(self, sentence : str, model,
        n_neighbors : int = 21,
        nprobe : int =1 ) :

        self.index.nprobe = nprobe

        encoding =  model.encode(sentence)
        encoding = encoding.reshape(1,-1)

        D, I = self.index.search(encoding, n_neighbors)


        result = {
            "id" : int(self.labels[I[:,0]]),
            "label" : config.target_name_dict[int(self.labels[I[:,0]])],
            'distance': float(D[:,0]) 
        }
        return result


# if __name__ == "__main__" : 
#     from sklearn.datasets import fetch_20newsgroups
#     from sentence_transformers import SentenceTransformer

#     model = SentenceTransformer(config.MODEL_FOLDER)

#     test = fetch_20newsgroups(subset="test", remove=("headers", "footers","quotes"), shuffle=True)
#     data = fetch_20newsgroups(subset="train", remove=("headers", "footers","quotes"), shuffle=True)
#     labels = data.target
#     # Get Sentence_transformers model
#     with open(config.MATRIX_EMBEDDING_PATH, "rb") as f : 
#         matrix = pickle.load(f)
#     search_engine = SearchEngine(matrix, labels)

#     for i in test.data[:5] : 
#         print(i)
#         print(search_engine.predict(i, model = model))
#         print("*"*100)


#     with open(config.LABELS_PATH, "wb") as f : 
#         pickle.dump(labels, f)  