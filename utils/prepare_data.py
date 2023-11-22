import os.path
import shutil
import tqdm
from nltk.tokenize import word_tokenize

class PrepareData:

    def __init__(self, input_dir: str):
        self.input_dir = input_dir
        self.dataset_name = os.path.basename(input_dir)
        self.output_dir = os.path.join("data", self.dataset_name)

    def convert(self):
        self._convert_file("train")
        self._convert_file("dev")
        self._convert_file("test")

    def _convert_file(self, split: str):
        split_dir = os.path.join(self.output_dir, "val" if split == "dev" else split)
        os.makedirs(split_dir, exist_ok=True)
        input_path = os.path.join(self.input_dir, f"{split}.tsv")
        src_all, tgt_all = [], []
        with open(input_path, "r", encoding="utf-8") as input_file:
            for line in tqdm.tqdm(input_file):
                values = [val.strip() for val in line.split("\t")]
                assert len(values) == 2
                src = values[0]
                tgt = eval(values[1])[0] if split == "test" else values[1]
                src_all.append(self._norm(src))
                tgt_all.append(self._norm(tgt))
        with open(os.path.join(split_dir, "src.txt"), "w", encoding="utf-8") as output_file:
            for src in src_all:
                output_file.write(src)
                output_file.write("\n")
        with open(os.path.join(split_dir, "tgt.txt"), "w", encoding="utf-8") as output_file:
            for tgt in tgt_all:
                output_file.write(tgt)
                output_file.write("\n")
        shutil.copy(input_path, os.path.join(split_dir, os.path.basename(input_path)))

    def _norm(self, val: str):
        return " ".join(word_tokenize(val, preserve_line=False)).lower()


if __name__ == '__main__':
    for ds in ("cnn_news", "mscoco", "qqp"):
        prepare = PrepareData(ds)
        prepare.convert()
