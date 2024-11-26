from pathlib import Path
from tokenizers import BertWordPieceTokenizer

data_dir = Path("data/samples")
file_paths = [str(f) for f in data_dir.glob("*.txt")]

### training own tokenizer
tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=True
)

## train tokenizer
tokenizer.train( 
    files=file_paths,
    vocab_size=30_000, 
    min_frequency=5,
    limit_alphabet=1000, 
    wordpieces_prefix='##',
    special_tokens=['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
    )

## save tokenizer
output_dir = Path("models")
output_dir.mkdir(parents= True, exist_ok=True)
tokenizer.save_model(str(output_dir))
print("Tokenizer trained !!")








