Converts audio files (.wav, .mp3, .mp4) into a diarized Praat Textgrid using transcription from Whisper and diarization from Pyannote. 
This script was originally written to serve as a transcription pipeline for the MEET corpus created at KTH Royal Insitute of Technology.

In order to run this, you will need access to the precision-2 model. This requires a premium subscription to Pyannote, but as of February 2026 there is a free trial available for 1 month.

# Instructions

## Preliminary Setup

1. Download the repository and import the necessary packages
2. Create an account with Pyannote if you don't already have one and either start a free trial or buy a package that allows you access to their precision-2 model.
3. Create a Pyannote API token and save it somewhere

## How to Run

1. Make sure you have python installed and install the required packages with:
```bash
pip install -r requirements.txt
``` 
2. Next, you must upload the audio file to the Pyannote API. To do this, run the following:
```bash
python upload_audio.py <audio_file_path.wav> <pyannote_api_key>
```
Replace <audio_file_path.wav> with the path to the audio file you would like to transcribe, and replace <pyannote_api_key> with your API key you made in the preliminary setup.

3. Run the following:
```bash
python MEET_transcription_pipeline.py <audio_file_path.wav> <whisper_model_type> <pyannote_api_key> <num_speakers>
```
Replace <whisper_model_type> with the whisper model you would like to use. In order of increasing size: tiny, base, small, medium, turbo, large. 
For English only dialogue: tiny.en, base.en, small.en and medium.en are preferable for those models.
( For MEET corpus transcription use the large model).

Replace <num_speakers> with the number of speakers present in the audio file. If this is not known, you can write "unknown" here.
Example:
```bash
python MEET_transcription_pipeline.py audio_files/audio.wav large 123456 3
```

It may take a few minutes for the whisper transcription to run. Upon completion, a .txt file with the job ID for this pyannote diarization job will be created.
A TextGrid file is also created with the diarizaed transcription. This file is called "diarized_precision2_<file_name>.TextGrid"

4. If you want to run the script on the same audio file again, you can use the same pyannote API job as before.
Simply take the job ID that was saved in job_IDs.txt and add it as an argument in the same command:
```bash
python MEET_transcription_pipeline.py <audio_file_path.wav> <whisper_model_type> <pyannote_api_key> <num_speakers> <job_DI>
```

# Notes:

The script takes word-level timestamps from the whisper transcription and pairs them up with the closest matching pyannote diarization timestamps.
Then the words are stitched back together into Inter-pausal Units (defined by segments of speech by one speaker without a pause of longer than 500ms).

At the word level, Whisper has a tendency to timestamp some short words as having zero duration. 
If these words are left isolated after creating Inter-pausal Units, these words are artificially lengthed to 40ms, since a lot of software (e.g. ELAN) will simply ignore zero-duration entries otherwise.


