import hydra.utils
import bioc
from tqdm import tqdm


class BiocDataset:
    def __init__(self, path, tokenizer, limit_examples, limit_documents, use_doc_context,
                 mark_with_special_tokens, blind_entities, max_length,
                 entity_to_side_information, pair_to_side_information,
                 entity_to_embedding_index, use_none_class=False):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.meta = utils.get_dataset_metadata(path)
        self.use_none_class = use_none_class
        self.entity_to_side_information = entity_to_side_information
        self.pair_to_side_information = pair_to_side_information
        self.entity_to_embedding_index = entity_to_embedding_index
        with open(hydra.utils.to_absolute_path(path)) as f:
            collection = bioc.load(f)
            doc: bioc.BioCDocument
            for doc in tqdm(collection.documents[:limit_documents], desc="Loading data"):
                if limit_examples and len(self.examples) > limit_examples:
                    break
                for passage in doc.passages:
                    for sentence in passage.sentences:
                        if use_doc_context:
                            doc_context = doc.passages[0].text
                        else:
                            doc_context = None
                        self.examples += sentence_to_examples(
                            sentence,
                            tokenizer,
                            doc_context=doc_context,
                            pair_types=self.meta.pair_types,
                            label_to_id=self.meta.label_to_id,
                            mark_with_special_tokens=mark_with_special_tokens,
                            blind_entities=blind_entities,
                            max_length=self.max_length,
                            use_none_class=use_none_class,
                            entity_to_side_information=self.entity_to_side_information,
                            pair_to_side_information=self.pair_to_side_information,
                            entity_to_embedding_index=self.entity_to_embedding_index
                        )

                        if limit_examples and len(self.examples) > limit_examples:
                            break

    def __getitem__(self, item):
        example = self.examples[item]["features"].copy()
        example["meta"] = self.meta

        return example

    def __len__(self):
        return len(self.examples)