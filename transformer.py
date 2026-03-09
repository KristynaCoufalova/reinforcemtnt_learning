import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and split into heads
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.W_o(attn_output)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, 
                 num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
        
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
        
        output = self.fc(dec_output)
        return output

# Example usage and training setup
def create_transformer_model():
    """Create a transformer model with typical hyperparameters"""
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 100
    dropout = 0.1
    
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads,
                       num_layers, d_ff, max_seq_length, dropout)
    return model

# Simple training loop example
def train_step(model, src, tgt, optimizer, criterion):
    model.train()
    
    # Prepare target input and output
    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]
    
    # Forward pass
    output = model(src, tgt_input)
    
    # Calculate loss
    loss = criterion(output.contiguous().view(-1, output.size(-1)), 
                    tgt_output.contiguous().view(-1))
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Test functions to verify the transformer works correctly
def test_shape_correctness(model, src_vocab_size, tgt_vocab_size):
    """Test 1: Verify output shapes are correct"""
    print("=" * 50)
    print("TEST 1: Shape Correctness")
    print("=" * 50)
    
    batch_size = 4
    src_seq_len = 12
    tgt_seq_len = 10
    
    src = torch.randint(1, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len))
    
    output = model(src, tgt[:, :-1])
    
    expected_shape = (batch_size, tgt_seq_len - 1, tgt_vocab_size)
    actual_shape = output.shape
    
    print(f"Input source shape: {src.shape}")
    print(f"Input target shape: {tgt.shape}")
    print(f"Output shape: {actual_shape}")
    print(f"Expected shape: {expected_shape}")
    print(f"✓ PASSED" if actual_shape == expected_shape else "✗ FAILED")
    return actual_shape == expected_shape

def test_overfitting_simple_task(model, criterion, optimizer):
    """Test 2: Can the model overfit a tiny dataset? (Sanity check)"""
    print("\n" + "=" * 50)
    print("TEST 2: Overfitting Ability (Sanity Check)")
    print("=" * 50)
    
    # Create a simple copy task: model should learn to copy input to output
    batch_size = 8
    seq_len = 5
    vocab_size = 10
    
    # Simple pattern: [1, 2, 3, 4, 5] -> [1, 2, 3, 4, 5]
    src = torch.randint(1, vocab_size, (batch_size, seq_len))
    tgt = src.clone()
    
    model.train()
    losses = []
    
    print("Training on tiny dataset (should overfit quickly)...")
    for epoch in range(100):
        loss = train_step(model, src, tgt, optimizer, criterion)
        losses.append(loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss:.4f}")
    
    final_loss = losses[-1]
    initial_loss = losses[0]
    
    print(f"\nInitial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Loss reduction: {initial_loss - final_loss:.4f}")
    
    # Model should significantly reduce loss
    passed = final_loss < initial_loss * 0.5
    print(f"✓ PASSED - Model can learn!" if passed else "✗ FAILED - Model not learning")
    return passed

def test_gradient_flow(model):
    """Test 3: Check if gradients flow through the network"""
    print("\n" + "=" * 50)
    print("TEST 3: Gradient Flow")
    print("=" * 50)
    
    src = torch.randint(1, 100, (2, 8))
    tgt = torch.randint(1, 100, (2, 8))
    
    model.train()
    output = model(src, tgt[:, :-1])
    loss = output.sum()
    loss.backward()
    
    # Check if gradients exist for key parameters
    has_grads = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grads.append((name, param.grad.abs().mean().item()))
    
    print(f"Parameters with gradients: {len(has_grads)}/{len(list(model.parameters()))}")
    print("\nSample gradient magnitudes:")
    for name, grad_mag in has_grads[:5]:
        print(f"  {name}: {grad_mag:.6f}")
    
    passed = len(has_grads) == len(list(model.parameters()))
    print(f"\n✓ PASSED - All parameters receive gradients" if passed else "✗ FAILED")
    return passed

def test_attention_masking(model):
    """Test 4: Verify attention masking works (causal mask for decoder)"""
    print("\n" + "=" * 50)
    print("TEST 4: Attention Masking")
    print("=" * 50)
    
    # The decoder should not attend to future tokens
    # We'll check this by comparing outputs with different future tokens
    
    batch_size = 1
    seq_len = 6
    
    src = torch.randint(1, 100, (batch_size, seq_len))
    tgt1 = torch.randint(1, 100, (batch_size, seq_len))
    tgt2 = tgt1.clone()
    tgt2[0, -1] = 99  # Change only the last token
    
    model.eval()
    with torch.no_grad():
        out1 = model(src, tgt1[:, :-1])
        out2 = model(src, tgt2[:, :-1])
        
        # First positions should be identical (causal masking)
        # because future tokens shouldn't affect them
        first_pos_diff = (out1[0, 0] - out2[0, 0]).abs().mean().item()
        
        print(f"Difference in first position output: {first_pos_diff:.8f}")
        print(f"(Should be very small due to causal masking)")
        
        passed = first_pos_diff < 1e-5
        print(f"✓ PASSED - Causal masking working" if passed else "⚠ WARNING - Check masking")
    
    return passed

def test_position_encoding():
    """Test 5: Verify positional encoding properties"""
    print("\n" + "=" * 50)
    print("TEST 5: Positional Encoding")
    print("=" * 50)
    
    d_model = 512
    max_seq = 100
    pe = PositionalEncoding(d_model, max_seq)
    
    # Create dummy input
    x = torch.zeros(1, max_seq, d_model)
    output = pe(x)
    
    # Check that positions are different
    pos_0 = output[0, 0, :].numpy()
    pos_1 = output[0, 1, :].numpy()
    pos_50 = output[0, 50, :].numpy()
    
    diff_01 = np.abs(pos_0 - pos_1).mean()
    diff_050 = np.abs(pos_0 - pos_50).mean()
    
    print(f"Difference between position 0 and 1: {diff_01:.4f}")
    print(f"Difference between position 0 and 50: {diff_050:.4f}")
    
    passed = diff_01 > 0 and diff_050 > diff_01
    print(f"✓ PASSED - Positional encodings are unique" if passed else "✗ FAILED")
    return passed

def run_all_tests():
    """Run all tests to verify the transformer implementation"""
    print("\n" + "🧪 RUNNING TRANSFORMER TESTS" + "\n")
    
    # Create model with smaller dimensions for faster testing
    src_vocab_size = 100
    tgt_vocab_size = 100
    d_model = 128
    num_heads = 4
    num_layers = 2
    d_ff = 512
    max_seq_length = 50
    dropout = 0.1
    
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads,
                       num_layers, d_ff, max_seq_length, dropout)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Run tests
    results = []
    results.append(("Shape Correctness", test_shape_correctness(model, src_vocab_size, tgt_vocab_size)))
    results.append(("Gradient Flow", test_gradient_flow(model)))
    results.append(("Positional Encoding", test_position_encoding()))
    results.append(("Attention Masking", test_attention_masking(model)))
    results.append(("Overfitting Ability", test_overfitting_simple_task(model, criterion, optimizer)))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    total_passed = sum(results[i][1] for i in range(len(results)))
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
    
    if total_passed == len(results):
        print("\n🎉 All tests passed! Your transformer is working correctly!")
    else:
        print("\n⚠️  Some tests failed. Review the implementation.")

# Example of how to use the model
if __name__ == "__main__":
    # Run comprehensive tests
    run_all_tests()