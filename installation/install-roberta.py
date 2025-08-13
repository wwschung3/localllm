from transformers import AutoModelForQuestionAnswering, AutoTokenizer
model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
model.save_pretrained("./roberta_model")
tokenizer.save_pretrained("./roberta_model")