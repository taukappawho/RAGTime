from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-minilm-l12-v2")
print(tokenizer.model_max_length)