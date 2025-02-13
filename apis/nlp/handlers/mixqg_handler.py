import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained('t5-base')
model = AutoModelForSeq2SeqLM.from_pretrained('Salesforce/mixqg-base').to(device)

def format_inputs(context: str, answer: str):
    # return f"answer:{answer} context:{context}"
    return f"{answer} \\n {context}"

# batch_inputs = [
#     {"context": "The capital of France is Paris.", "answers": [France, Paris]}
# ]

def generate_question(batch_inputs):
    results = []
    for _input in batch_inputs:
        questions = []
        for answer in _input["answers"]:
            formatted_input = format_inputs(
                _input["context"],
                answer
            )
            input_ids = tokenizer(formatted_input, return_tensors="pt", padding='longest',
                                              truncation=True, max_length=1024).input_ids.to(device)
            generated_ids = model.generate(input_ids, max_length=32, num_beams=4)
            output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            questions.extend(output)
        results.append(questions)
    return results
        