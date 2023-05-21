from typing import Dict, Optional, Union

import numpy as np
from scipy.io.wavfile import write as write_wav

from .generation import codec_decode, generate_coarse, generate_fine, generate_text_semantic, \
    SAMPLE_RATE, generate_coarse_stream


def text_to_semantic(
        text: str,
        history_prompt: Optional[Union[Dict, str]] = None,
        temp: float = 0.7,
        silent: bool = False,
):
    """Generate semantic array from text.

    Args:
        text: text to be turned into audio
        history_prompt: history choice for audio cloning
        temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar

    Returns:
        numpy semantic array to be fed into `semantic_to_waveform`
    """
    x_semantic = generate_text_semantic(
        text,
        history_prompt=history_prompt,
        temp=temp,
        silent=silent,
        use_kv_caching=True
    )
    return x_semantic


def semantic_to_waveform(
        semantic_tokens: np.ndarray,
        history_prompt: Optional[Union[Dict, str]] = None,
        temp: float = 0.7,
        silent: bool = False,
        output_full: bool = False,
):
    """Generate audio array from semantic input.

    Args:
        semantic_tokens: semantic token output from `text_to_semantic`
        history_prompt: history choice for audio cloning
        temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar
        output_full: return full generation to be used as a history prompt

    Returns:
        numpy audio array at sample frequency 24khz
    """
    coarse_tokens = generate_coarse(
        semantic_tokens,
        history_prompt=history_prompt,
        temp=temp,
        silent=silent,
        use_kv_caching=True
    )
    fine_tokens = generate_fine(
        coarse_tokens,
        history_prompt=history_prompt,
        temp=0.5,
    )
    audio_arr = codec_decode(fine_tokens)
    if output_full:
        full_generation = {
            "semantic_prompt": semantic_tokens,
            "coarse_prompt": coarse_tokens,
            "fine_prompt": fine_tokens,
        }
        return full_generation, audio_arr
    return audio_arr


def save_as_prompt(filepath, full_generation):
    assert (filepath.endswith(".npz"))
    assert (isinstance(full_generation, dict))
    assert ("semantic_prompt" in full_generation)
    assert ("coarse_prompt" in full_generation)
    assert ("fine_prompt" in full_generation)
    np.savez(filepath, **full_generation)


def generate_audio(
        text: str,
        history_prompt: Optional[Union[Dict, str]] = None,
        text_temp: float = 1.0,
        waveform_temp: float = 1.0,
        silent: bool = False,
        output_full: bool = False,
):
    """Generate audio array from input text.

    Args:
        text: text to be turned into audio
        history_prompt: history choice for audio cloning
        text_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        waveform_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar
        output_full: return full generation to be used as a history prompt

    Returns:
        numpy audio array at sample frequency 24khz
    """
    semantic_tokens = text_to_semantic(
        text,
        history_prompt=history_prompt,
        temp=text_temp,
        silent=silent,
    )
    out = semantic_to_waveform(
        semantic_tokens,
        history_prompt=history_prompt,
        temp=waveform_temp,
        silent=silent,
        output_full=output_full,
    )
    if output_full:
        full_generation, audio_arr = out
        return full_generation, audio_arr
    else:
        audio_arr = out
    return audio_arr


def generate_audio_stream(
        text: str,
        history_prompt: Optional[Union[Dict, str]] = None,
        text_temp: float = 1.0,
        waveform_temp: float = 1.0,
        silent: bool = False,
        sliding_window_len: int = 60
):
    x_semantic = text_to_semantic(
        text,
        history_prompt=history_prompt,
        temp=text_temp,
        silent=silent,
    )
    previous_coarse_size = 0
    fine_tokens = None
    for coarse_tokens in generate_coarse_stream(
            x_semantic,
            history_prompt=history_prompt,
            temp=waveform_temp,
            silent=silent,
            use_kv_caching=True,
            sliding_window_len=sliding_window_len
    ):
        coarse_tokens = np.array(coarse_tokens)
        coarse_tokens_cropped = coarse_tokens[:, previous_coarse_size:]
        previous_coarse_size = coarse_tokens.shape[1]
        batch_fine_tokens = generate_fine(
            coarse_tokens_cropped,
            history_prompt=history_prompt,
            temp=0.5,
        )
        if fine_tokens is None:
            fine_tokens = batch_fine_tokens
        else:
            fine_tokens = np.concatenate([fine_tokens, batch_fine_tokens], axis=1)
        # audio_arr = codec_decode(batch_fine_tokens)
        # write_wav(f"bark_generation_{counter}.wav", SAMPLE_RATE, audio_arr)
        # counter += 1
    audio_arr = codec_decode(fine_tokens)
    write_wav(f"bark_generation_merged.wav", SAMPLE_RATE, audio_arr)
    final_fine_tokens = generate_fine(
        coarse_tokens,
        history_prompt=history_prompt,
        temp=0.5,
    )
    final_audio_arr = codec_decode(final_fine_tokens)
    return final_audio_arr
