
import os

from numpy import matrix
import config
import pickle
import logging
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
from pipeline import SearchEngine
from sklearn.datasets import fetch_20newsgroups

class Benchmark(SearchEngine) : 

    def __init__(self, test_data, test_labels, model, matrix, labels,
                    nlist = 100, embedding_size = 768) : 
        
        SearchEngine.__init__(self, 
        matrix = matrix,
        labels = labels,
        nlist = nlist, 
        embedding_size = embedding_size)

        self.test_data = test_data
        self.test_labels = test_labels
        self.model = model
        
    def benchmark(self) : 

        data = fetch_20newsgroups(subset="train", remove=("headers", "footers","quotes"), shuffle=True)
        labels = data.target

        #self.matrix = self.model.encode(self.train)
        results = []
        for line in self.test_data : 
            results.append(self.predict(line, self.model))

        res = [x["id"] for x in results]

        return classification_report(self.test_labels, res)


if __name__ == "__main__" : 

    if not os.path.exists(config.MATRIX_EMBEDDING_PATH) : 

        os.mkdir("train_embedding")
        
        # Import data 
        data = fetch_20newsgroups(subset="train", remove=("headers", "footers","quotes"), shuffle=True)

        train = data.data
        labels = data.target

        # Instanciate Sentence transformer
        model = SentenceTransformer(config.MODEL_FOLDER)
        #logging.info("Compute train embedding")
        matrix = model.encode(train)
        with open(config.MATRIX_EMBEDDING_PATH, "wb") as f : 
            pickle.dump(matrix, f)

    else : 
        #logging.info("Loading matrix embedding")
        with open(config.MATRIX_EMBEDDING_PATH, "rb") as f : 
            matrix = pickle.load(f)

    data = fetch_20newsgroups(subset="train", remove=("headers", "footers","quotes"), shuffle=True)
    labels = data.target


    test = fetch_20newsgroups(subset="test", remove=("headers", "footers","quotes"), shuffle=True)
    
    # Get Sentence_transformers model
    #logging.info("Pipeline benchmark with the test set")
    model = SentenceTransformer(config.MODEL_FOLDER)
    bm = Benchmark(
        test_data= test.data,
        test_labels=test.target,
        model = model,
        matrix = matrix,
        labels = labels
    )

    # Compute benchmark
    print(bm.benchmark())


