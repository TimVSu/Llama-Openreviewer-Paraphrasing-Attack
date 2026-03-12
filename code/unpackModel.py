from transformers import AutoModel, AutoTokenizer

model_id = "maxidl/Llama-OpenReviewer-8B"

model = AutoModel.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.save_pretrained("./my_model_ready")
tokenizer.save_pretrained("./my_model_ready")
