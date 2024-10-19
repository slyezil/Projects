from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained model tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained model
model = BertModel.from_pretrained('bert-base-uncased')

# Text to be tokenized
text = "Subhash Can do it !!"

# Encode text
input_ids = tokenizer.encode(text, add_special_tokens=True)

# Output the token IDs
print("Token IDs:", input_ids)

# Convert token IDs back to raw tokens and output them
raw_tokens = [tokenizer.decode([token_id]) for token_id in input_ids]
print("Raw tokens:", raw_tokens)

# Convert list of IDs to a tensor
input_ids_tensor = torch.tensor([input_ids])

# Pass the input through the model
with torch.no_grad():
    outputs = model(input_ids_tensor)

# Extract the embeddings
embeddings = outputs.last_hidden_state

# Print the embeddings
print("Embeddings: ", embeddings)
num_elements = embeddings.numel()
print(num_elements)