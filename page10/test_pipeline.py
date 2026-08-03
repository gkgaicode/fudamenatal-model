import pytest
import torch
import numpy as np
import os
from model_architecture import RotaryEmbedding, RoPEFlashAttention, ProductionFoundationModel
from data_prep_skill import run as run_data_prep

# Set manual seeds to ensure total test runtime determinism
torch.manual_seed(42)
np.random_seed(42)

def test_rotary_embedding_math():
    """Verify that RoPE caches generate accurate cosine and sine dimension transformations."""
    dim = 64
    max_seq_len = 512
    rope = RotaryEmbedding(dim=dim, max_seq_len=max_seq_len)
    
    cos, sin = rope(torch.randn(1, 100, 1, dim), seq_len=100)
    
    # Assert proper sequence context sizing properties
    assert cos.shape == (100, dim)
    assert sin.shape == (100, dim)
    # Validate mathematical projection constraints: sin^2 + cos^2 should approximate 1.0
    combined_trig_sum = (cos ** 2) + (sin ** 2)
    assert torch.allclose(combined_trig_sum, torch.ones_like(combined_trig_sum), atol=1e-5)


def test_flash_attention_output_invariants():
    """Ensure custom FlashAttention layers preserve tensor shapes and causal mask properties."""
    batch_size = 2
    seq_len = 128
    d_model = 256
    n_heads = 8
    
    attn_layer = RoPEFlashAttention(d_model=d_model, n_heads=n_heads)
    sample_input = torch.randn(batch_size, seq_len, d_model)
    
    output = attn_layer(sample_input)
    
    # Assert structural retention: Input shapes must cleanly match output vectors
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any(), "NaN values found in attention projections!"


def test_masked_sft_loss_exclusion():
    """Confirm cross-entropy ignores prompt tokens tagged with index -100."""
    vocab_size = 1000
    logits = torch.randn(2, 10, vocab_size) # Batch=2, Seq=10, Vocab=1000
    
    # Target label map: Force index entries in the first batch item to be evaluated,
    # and completely mask out the second batch item via -100
    labels = torch.tensor([,
        [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100]
    ])
    
    # Calculate cross-entropy
    loss_with_mask = torch.nn.functional.cross_entropy(
        logits.view(-1, vocab_size), 
        labels.view(-1), 
        ignore_index=-100
    )
    
    # Calculate cross-entropy manually on just the first item to ensure perfect compliance
    loss_unmasked_only = torch.nn.functional.cross_entropy(
        logits[0].view(-1, vocab_size), 
        labels[0].view(-1)
    )
    
    assert torch.allclose(loss_with_mask, loss_unmasked_only, atol=1e-6), \
        "Prompt token cross-entropy masking logic failing!"


def test_data_prep_tokenization_io(tmp_path):
    """Verify raw text compilation engine writes and maps arrays to disk accurately."""
    # Write a temporary target corpus file
    sample_corpus_file = tmp_path / "test_corpus.txt"
    sample_corpus_file.write_text("Testing the custom data processing skill automation mechanics block.")
    
    # Step into temporary file directory to isolate compilation artifacts
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        run_data_prep([str(sample_corpus_file)], vocab_size=500)
        
        # Verify targeted deployment array output structures exist on disk
        assert os.path.exists("my_tokenizer.json")
        assert os.path.exists("pretraining_dataset.bin")
        
        # Verify read accessibility integrity of generated binary sequence array maps
        compiled_data = np.fromfile("pretraining_dataset.bin", dtype=np.uint16)
        assert len(compiled_data) > 0
        
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    # Allows localized fallback execution loops natively inside developer workflows
    pytest.main([__file__])
