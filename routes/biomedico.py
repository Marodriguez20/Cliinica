from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("PlanTL-GOB-ES/roberta-base-biomedical-clinical-es")
model = AutoModel.from_pretrained("PlanTL-GOB-ES/roberta-base-biomedical-clinical-es")

text = "El paciente presenta síntomas compatibles con insuficiencia renal aguda."
inputs = tokenizer(text, return_tensors="pt")

outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
