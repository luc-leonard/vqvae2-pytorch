from perceiver.adapters import TextInputAdapter, TextOutputAdapter, TextMasking
from perceiver.model import PerceiverEncoder, PerceiverDecoder, PerceiverMLM


def create_encoder(max_seq_len, num_latent_channels,
                   num_encoder_layers, num_encoder_cross_attention_heads,
                   dropout, vocab_size, num_encoder_self_attention_layers_per_block
                   , latent_shape):
    input_adapter = TextInputAdapter(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        num_input_channels=num_latent_channels)
    encoder = PerceiverEncoder(
        input_adapter=input_adapter,
        latent_shape=latent_shape,
        num_layers=num_encoder_layers,
        num_cross_attention_heads=num_encoder_cross_attention_heads,
        num_self_attention_heads=num_encoder_cross_attention_heads,
        num_self_attention_layers_per_block=num_encoder_self_attention_layers_per_block,
        dropout=dropout)
    return encoder


def create_model(num_latents, num_latent_channels, vocab_size, max_seq_len,
                 num_decoder_cross_attention_heads, dropout, num_encoder_layers,
                 num_encoder_cross_attention_heads, num_encoder_self_attention_layers_per_block):
    latent_shape = (num_latents,
                    num_latent_channels)
    encoder = create_encoder(max_seq_len,
                             num_latent_channels,
                             num_encoder_layers,
                             num_encoder_cross_attention_heads,
                             dropout,
                             vocab_size,
                             num_encoder_self_attention_layers_per_block,
                             latent_shape)
    output_adapter = TextOutputAdapter(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        num_output_channels=num_latent_channels)
    decoder = PerceiverDecoder(
        output_adapter=output_adapter,
        latent_shape=latent_shape,
        num_cross_attention_heads=num_decoder_cross_attention_heads,
        dropout=dropout)
    return PerceiverMLM(encoder, decoder, TextMasking(vocab_size))
