Implement the DCN v2 for binary classification problems
```
/project
    ├── data_preprocessing.py
    ├── model.py
    ├── train.py
    ├── hyperparameter_optimization.py
    ├── shap_analysis.py
    ├── utils.py
    └── main.py
```

INPUT:
  ├─ Numerical features         (num_numerical)
  ├─ Categorical indices        (categorical_emb: for embedding)
  └─ One-hot encoded features   (categorical_onehot)

↓
[1] EMBEDDING LAYER (for categorical_emb)
    ├─ nn.Embedding(cardinality + 1, embedding_dim) per categorical feature
    └─ Concatenate all embeddings → [batch_size, emb_total_dim]

↓
[2] CONCATENATE INPUTS:
    ├─ [numerical, all embeddings, one-hot features]
    └─ Result: input vector x of shape [batch_size, input_dim]

↓
[3] CROSS NETWORK (CrossLayerV2 × N layers)
    ├─ Each layer receives (x0, xl)
    ├─ Computes: x₀ * V(U(xl)) + bias + xl
    └─ Output: cross_output (same shape as input)

↓
[4] DEEP NETWORK (Feedforward MLP)
    ├─ Layers: Linear → ReLU → Linear → ReLU → ...
    └─ Output: deep_output of shape [batch_size, deep_layers[-1]]

↓
[5] CONCATENATE cross_output and deep_output
    └─ Shape: [batch_size, input_dim + deep_output_dim]

↓
[6] FINAL OUTPUT LAYER
    ├─ Linear(input_dim + deep_output_dim → 1)
    └─ Activation: Sigmoid
    → Binary classification prediction

OUTPUT:
    └─ Predicted probability per sample (shape: [batch_size])
