from transformers import AutoTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset

# Define the model name
model_name = "google/flan-t5-small"

# Load the tokenizer and model
pretrained_tokenizer = AutoTokenizer.from_pretrained(model_name)
pretrained_model = T5ForConditionalGeneration.from_pretrained(model_name)

# Assign the eos_token as the pad_token
pretrained_tokenizer.pad_token = pretrained_tokenizer.eos_token

# Load the Taylor Swift dataset
dataset = load_dataset("lamini/taylor_swift")

# Split into training and test sets
train_test_split = dataset['train'].train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Evaluate the model's performance
def generate_response(input_text, model, tokenizer, max_length=512):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=False, max_length=512)

    # Generate the response with adjusted parameters
    output_sequences = model.generate(
        input_ids=inputs['input_ids'],
        max_length=max_length,
        repetition_penalty=1.2,  # Apply repetition penalty
        no_repeat_ngram_size=2,  # Prevent repetition of 2-grams
        do_sample=True,          # Enable sampling for non-deterministic output
        temperature=0.7,         # Control randomness (lower is less random)
        top_k=50,                # Consider top 50 tokens for sampling
        top_p=0.95               # Consider tokens with cumulative probability 0.95
    )

    # Decode and return the response
    return tokenizer.decode(output_sequences[0], skip_special_tokens=True)

# Test the pre-trained model
print("################################################\n")
print("Testing pre-trained model:\n")
print("################################################\n")
for x in range(0, 3):
    sample_question = train_dataset[x]['question']
    sample_answer = train_dataset[x]['answer']
    print(f"\n\nQuestion: {sample_question}\n")
    print(f"Generated Answer: {generate_response(sample_question, pretrained_model, pretrained_tokenizer )}\n")
    print(f"Expected Answer: {sample_answer}\n\n")


# Tokenize data
def preprocess_function(examples):

    # Tokenize the input (question)
    model_inputs = pretrained_tokenizer(examples["question"], max_length=512, truncation=True, padding='max_length')

    # Tokenize the target (answer) using text_target
    labels = pretrained_tokenizer(examples["answer"], max_length=512, truncation=True, padding='max_length', return_tensors="pt").input_ids
    
    # Important: Replace padding token id with -100 to ignore in loss calculation
    labels = [[(label if label != pretrained_tokenizer.pad_token_id else -100) for label in label_example] for label_example in labels]
    
    model_inputs["labels"] = labels
    return model_inputs

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

# Define the training parameters
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",  # Updated from "evaluation_strategy" to "eval_strategy"
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Set up the Trainer class with the model, training arguments, and datasets.
trainer = Trainer(
    model=pretrained_model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset
)

# Fine-tune the model with the training data.
trainer.train()

# Assess the model's performance on the test dataset.
results = trainer.evaluate()
print(f"Test Results: {results}")

# Save the fine-tuned model and tokenizer
pretrained_model.save_pretrained('./fine-tuned-taylor-swift')
pretrained_tokenizer.save_pretrained('./fine-tuned-taylor-swift')

# Load the fine-tuned model and tokenizer
finetuned_Tokenizer = AutoTokenizer.from_pretrained('./fine-tuned-taylor-swift')
finetuned_Model = T5ForConditionalGeneration.from_pretrained('./fine-tuned-taylor-swift')

# Evaluate the model's performance after fine-tuning to see improvements.
print("################################################\n")
print("Testing fine-tuned model:\n")
print("################################################\n")
for x in range(0, 3):
    
    sample_question = train_dataset[x]['question']
    sample_answer = train_dataset[x]['answer']

    print(f"\n\nQuestion: {sample_question}\n")
    print(f"Generated Answer: {generate_response(sample_question, finetuned_Model, finetuned_Tokenizer )}\n")
    print(f"Expected Answer: {sample_answer}\n\n")

