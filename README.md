# Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset
Build a Named Entity Recognition (NER) model that can automatically identify and classify entities like names of people, locations, organizations, and other important terms from text. The goal is to tag each word in a sentence with its corresponding entity label.

## DESIGN STEPS
STEP 1
Import necessary libraries and set up the device (CPU or GPU).

STEP 2
Load the NER dataset and fill missing values.

STEP 3
Create word and tag dictionaries for encoding.

STEP 4
Group words into sentences and encode them into numbers.

STEP 5
Build a BiLSTM model for sequence tagging.

STEP 6
Train the model using the training data.

STEP 7
Evaluate the model performance on test data.

## PROGRAM
### Name:SELVARANI S
### Register Number:212224040301
```python
class BiLSTMTagger(nn.Module):
  def __init__(self, vocab_size, tagset_size, embedding_dim = 50, hidden_dim = 100):
    super(BiLSTMTagger, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.dropout = nn.Dropout(0,1)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
    self.fc = nn.Linear(hidden_dim * 2, tagset_size)

  def forward(self, x):
    x = self.embedding(x)
    x = self.dropout(x)
    x, _ = self.lstm(x)
    return self.fc(x)
```        
```
model = BiLSTMTagger(len(word2idx)+1, len(tag2idx)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

```
```
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
      model.train()
      total_loss = 0
      for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = loss_fn(outputs.view(-1, len(tag2idx)), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
      train_losses.append(total_loss)

      model.eval()
      val_loss = 0
      with torch.no_grad():
        for batch in test_loader:
          input_ids = batch["input_ids"].to(device)
          labels = batch['labels'].to(device)
          outputs = model(input_ids)
          loss = loss_fn(outputs.view(-1, len(tag2idx)), labels.view(-1))
          val_loss += loss.item()
      val_losses.append(val_loss)
      print(f"Epoch {epoch+1}: Train Loss = {total_loss:.4f}, Val Loss = {val_loss:.4f}")          

    return train_losses, val_losses


```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
<img width="1011" height="150" alt="image" src="https://github.com/user-attachments/assets/dcbe91d0-93d6-4fb7-a958-082b71b79ffa" />



<img width="761" height="548" alt="image" src="https://github.com/user-attachments/assets/8a6a4668-b5c1-4640-aba2-5060edd8ce1e" />



### Sample Text Prediction
<img width="853" height="817" alt="image" src="https://github.com/user-attachments/assets/d62d8fe6-c958-45ba-a784-6894beea303a" />

## RESULT
The BiLSTM NER model achieved good accuracy in identifying entities like persons, locations, and organizations. It showed strong performance on frequent tags, with scope for improvement on rarer ones.
