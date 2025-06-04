
import numpy as np
class Retriever:
    def __init__(
        self,
        model,
        add_prefix=False,
        query_prefix="Represent this query for searching relevant code",
        document_prefix="",
        normalize=True,
        binarize=False,
        multiprocess = True
    ):
        self.model = model
        self.multiprocess = multiprocess
        if multiprocess:
            self.gpu_pool = self.model.start_multi_process_pool()
        self.add_prefix = add_prefix
        self.doc_as_query = False
        self.query_prefix = query_prefix
        self.docoment_prefix = document_prefix
        self.normalize = normalize
        self.binarize = binarize

    def set_normalize(self, normalize):
        self.normalize = normalize

    def encode(self, sentences, **kwargs):
        if self.add_prefix:
            print(f"Adding prefix: {self.query_prefix}")
            sentences = [f"{self.query_prefix}: {sent}" for sent in sentences]
        kwargs.pop('convert_to_tensor')
        kwargs["normalize_embeddings"] = self.normalize
        return self.model.encode_multi_process(sentences, self.gpu_pool, **kwargs)

    def encode_queries(self, queries, **kwargs) -> np.ndarray:
        if self.add_prefix and self.query_prefix != "":
            input_texts = [f'{self.query_prefix}: {q}' for q in queries]
        else:
            input_texts = queries

        kwargs.pop('convert_to_tensor')
        kwargs["normalize_embeddings"] = self.normalize
        if self.multiprocess:
            return self.model.encode_multi_process(input_texts, self.gpu_pool, **kwargs)
        return self.model.encode(input_texts, **kwargs)

    def encode_corpus(self, corpus, **kwargs) -> np.ndarray:
        if isinstance(corpus[0], dict):
            input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        else:
            input_texts = corpus
        if self.add_prefix:
            if self.doc_as_query and self.query_prefix != "":
                input_texts = [f'{self.query_prefix}: {t}' for t in input_texts]
            elif self.docoment_prefix != "":
                input_texts = [f'{self.docoment_prefix}: {t}' for t in input_texts]

        kwargs.pop('convert_to_tensor')
        kwargs["normalize_embeddings"] = self.normalize
        if self.multiprocess:
            return self.model.encode_multi_process(input_texts, self.gpu_pool, **kwargs)
        return self.model.encode(input_texts, **kwargs)