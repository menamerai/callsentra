import datetime
import json
from pathlib import Path
import whisperx

def get_token(): 
    import os
    from dotenv import load_dotenv
    load_dotenv()
    return os.environ.get("HF_TOKEN")

def raw_diarize(
    hf_token: str,
    audio_path: str,
    output_path: str | None = None,
    device: str = "cpu",
    batch_size: int = 16,
    compute_type: str = "float16",
    whisper_model: str = "medium.en",
) -> list[dict[str, str | float | list[dict[str, str | float]]]]:
    """
    Produce the raw diarization pipeline output.

    Args:
    - hf_token: The Hugging Face API token. You can get it from your Hugging Face account. Make sure to get access to the pyannote spaces.
    - audio_path: The path to the audio file. e.g. "./audio/phone_call.mp3"
    - output_path: The path to the output file. e.g. "./output/phone_call.json"
    - device: The device to run the model on. e.g. "cuda" or "cpu"
    - batch_size: The batch size to use for the model. e.g. 16
    - compute_type: The compute type to use for the model. e.g. "float16" or "float32" or "int8"
    - whisper_model: The model to use for the whisperx. e.g. "medium.en"

    Returns:
    - A list of dictionaries containing the raw diarization pipeline output. It will be in the form of
    [
        {
            "start": 0.0,
            "end": 1.0,
            "speaker": "speaker_00",
            text: "Hello world"
            words: [
                {
                    "start": 0.0,
                    "end": 0.5,
                    "word": "Hello",
                    "speaker": "speaker_00",
                    score: 0.9
                },
                {
                    "start": 0.5,
                    "end": 1.0,
                    "word": "world",
                    "speaker": "speaker_00",
                    score: 0.9
                }
            ]
        },
        ...
    ]
    """

    if not output_path:
        # use current time to generate output path
        output_path = f"./output/audio-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"

    # initial transcription
    model = whisperx.load_model(
        whisper_arch=whisper_model, device=device, compute_type=compute_type
    )

    audio = whisperx.load_audio(audio_path)
    raw_transcript = model.transcribe(audio, batch_size=batch_size)

    # align the transcription
    model_a, meta_a = whisperx.load_align_model(
        language_code=raw_transcript["language"], device=device
    )
    aligned_transcript = whisperx.align(
        raw_transcript["segments"],
        model_a,
        meta_a,
        audio,
        device,
        return_char_alignments=False,
    )

    # diarize the aligned transcription
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
    diarize_segments = diarize_model(audio)
    dia_transcript = whisperx.assign_word_speakers(diarize_segments, aligned_transcript)

    # save the output to a file
    with open(output_path, "w") as f:
        json.dump(dia_transcript["segments"], f, indent=4, sort_keys=True)

    return dia_transcript["segments"]


if __name__ == "__main__":
    # TODO: Remove this block when demo is done being developed
    import os

    from dotenv import load_dotenv

    load_dotenv()
    hf_token = get_token()

    # audio_path = Path("./audio/sample2.mp3")
    audio_path = "sample2.mp3"
    output_path = "./output/sample2.json"
    device = "cuda"
    batch_size = 16
    compute_type = "float16"
    whisper_model = "medium.en"

    print(
        raw_diarize(
            hf_token=hf_token,
            audio_path=audio_path,
            output_path=output_path,
            device=device,
            batch_size=batch_size,
            compute_type=compute_type,
            whisper_model=whisper_model,
        )
    )
