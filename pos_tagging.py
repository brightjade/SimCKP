import os.path as osp
import json
from nltk.parse import CoreNLPParser
from tqdm import tqdm

word_tokenizer = CoreNLPParser(url="http://localhost:9000")
pos_tagger = CoreNLPParser(url="http://localhost:9000", tagtype='pos')

### Process data ###
for dataset in ["inspec", "krapivin", "nus", "semeval", "kp20k"]:
    data_dir = f"data/{dataset}"
    splits = ["train", "valid", "test"] if dataset == "kp20k" else ["test"]
    for split in splits:
        with open(osp.join(data_dir, f"{split}_src.txt")) as src_f, \
             open(osp.join(data_dir, f"{split}_trg.txt")) as trg_f, \
             open(osp.join(data_dir, f"{split}_postagged.json"), "w") as f:
            for i, (src_line, trg_line) in enumerate(tqdm(zip(src_f, trg_f))):
                title_and_context = src_line.strip().split("<eos>")
                if len(title_and_context) == 1:  # no title
                    title = ""
                    [context] = title_and_context
                elif len(title_and_context) == 2:
                    [title, context] = title_and_context
                else:
                    raise ValueError("The source text contains more than one title")
                title_words = list(word_tokenizer.tokenize(title.strip()))
                context_words = list(word_tokenizer.tokenize(context.strip()))
                src_words = title_words + context_words
                src_word_tag_pairs = pos_tagger.tag(src_words)
                _dict = {
                    "src_words": title_words + ["<eos>"] + context_words,
                    "src_word_tag_pairs": src_word_tag_pairs,
                    "src": src_line,
                    "trg": trg_line
                }
                json.dump(_dict, f)
                f.write("\n")
