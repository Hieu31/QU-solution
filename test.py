from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Thử với ViT5 nguyên bản (chưa qua Pilot)
model_name = "VietAI/vit5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

queries = ["d pasteur q3", "bv cho ray", "tahnhf pho ho chi minh"]

for q in queries:
    inputs = tokenizer(q, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    print(f"Input: {q}  -->  ViT5 Raw Output: {tokenizer.decode(outputs[0], skip_special_tokens=False)}")
