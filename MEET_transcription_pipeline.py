import requests
import os
import sys
import whisper
import numpy as np
import pandas as pd
import time
from collections import defaultdict

"""Run code in the format: script.py file_path model_type pyannote_key job_id. job_id can be omitted if you want to 
create a job."""

class MEET_Transcription_pipeline():
    def __init__(self, file_path, model_type, api_key, seperate_words = False):
        """ File path is the path to the audio file to be transcribed. Model type is the Whisper model to be used.
        The names of the models are: tiny.en, base.en, small.en, medium.en, turbo and large. For our purposes, medium
        is the optimal model, and tiny can be used for testing since it takes the least time."""

        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.model_type = model_type
        self.session_name = self.file_name.split(".")[0]
        self.model_name = model_type.split(".")[0]
        self.api_key = api_key
        self.seperate_words = seperate_words
    
    def whisper_transcription(self):
        
        model = whisper.load_model(self.model_type)
        result = model.transcribe(self.file_path, word_timestamps=True)
        self.text = result["text"]
        self.segments = result["segments"]
        self.audio_duration = self.segments[-1]["end"]

    def create_diarization_job(self):
        url = "https://api.pyannote.ai/v1/diarize"

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
                "url": f"media://{self.session_name}",
                "webhook": "https://example.com/webhook",
                "model": "precision-2",
                "numSpeakers": 3,
                "turnLevelConfidence": False,
                "exclusive": False,
                "confidence": False,
                "transcription": False
        }

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code != 200:
            raise TimeoutError(f"Error: {response.status_code} - {response.text}")
        else:
            print(response.json())
            return response.json()["jobId"]

    def poll_diarization(self, job_id):

        headers = {"Authorization": f"Bearer {self.api_key}"}

        while True:
            response = requests.get(
                f"https://api.pyannote.ai/v1/jobs/{job_id}", headers=headers
            )

            if response.status_code != 200:
                print(f"Error: {response.status_code} - {response.text}")
                break

            data = response.json()
            status = data["status"]

            if status in ["succeeded", "failed", "canceled"]:
                if status == "succeeded":
                    print("Job completed successfully!")
                    self.diarization = data["output"]["diarization"]
                else:
                    print(f"Job {status}")
                break

            print(f"Job status: {status}, waiting...")
    
    def align_outputs(self):

        # Assuming diarization_segments is a list of dictionaries from Step 1
        # Example: diarization_segments = [{"start": 0.5, "end": 3.2, "speaker": "SPEAKER_00"}, ...]
        diarize_df = pd.DataFrame(self.diarization)

        # Assuming transcript_result is a dictionary from Step 2
        # Example: transcript_result = {"segments": [{"start": 0.0, "end": 5.2, "text": "..."}, ...]}
        word_segments = []
        for sentence in self.segments:
            for word in sentence["words"]:
                word_segments.append(word)

        #transcript_segments = self.segments

        # If True, assign speakers even when there's no direct time overlap
        fill_nearest = True

        for seg in word_segments:

            intersections = np.minimum(diarize_df['end'], seg['end']) - \
                            np.maximum(diarize_df['start'], seg['start'])

            diarize_df_tmp = diarize_df.copy()
            diarize_df_tmp['intersection'] = intersections

            # keep only overlapping segments
            overlaps = diarize_df_tmp[diarize_df_tmp['intersection'] > 0]

            if not overlaps.empty:
                # assign speaker with max total overlap
                speaker = (
                    overlaps.groupby("speaker")["intersection"]
                    .sum()
                    .idxmax()
                )
            elif fill_nearest:
                # fallback to nearest diarization segment
                diarize_df_tmp['distance'] = np.minimum(
                    abs(diarize_df_tmp['start'] - seg['end']),
                    abs(diarize_df_tmp['end'] - seg['start'])
                )
                speaker = diarize_df_tmp.loc[diarize_df_tmp['distance'].idxmin(), "speaker"]
            else:
                speaker = None

            seg["speaker"] = speaker

        self.word_segments = word_segments
        return word_segments

    def stitch_sentences(self, expand_zero_durations = True):
        starting_segment = self.word_segments[0]
        current_speaker = starting_segment["speaker"]
        current_time = starting_segment["end"]
        stitched_transcription = []
        sentence = {"text": starting_segment["word"], "start": starting_segment["start"], "end": starting_segment["end"], "speaker": starting_segment["speaker"]}

        for word in self.word_segments[1:]:
            if word["speaker"] == current_speaker:
                if word["start"] <= 0.5 + current_time:
                    sentence["text"] += word["word"]
                    sentence["end"] = word["end"]
                    current_time = word["end"]
                    continue
            stitched_transcription.append(sentence)
            sentence = {"text": word["word"], "start": word["start"], "end": word["end"], "speaker": word["speaker"]}
            current_speaker = word["speaker"]
            current_time = word["end"]

        stitched_transcription.append(sentence)
        
        # Artificially expand zero-duration intervals
        if expand_zero_durations:
            for sentence in stitched_transcription:
                if sentence["start"] == sentence["end"]:
                    sentence["start"] = max(0, sentence["start"] - 0.02)
                    sentence["end"] = min(self.audio_duration, sentence["end"] + 0.02)


        return stitched_transcription
    
    def write_diarized_textgrid(self, segments, output_path, speaker_order=None):

        # Collect speakers
        speakers = sorted(set(seg["speaker"] for seg in segments))
        if speaker_order:
            speakers = speaker_order

        # Group segments by speaker
        speaker_segments = defaultdict(list)
        for seg in segments:
            speaker_segments[seg["speaker"]].append(seg)

        # Sort segments per speaker
        for spk in speakers:
            speaker_segments[spk].sort(key=lambda x: x["start"])

        lines = []
        lines.append('File type = "ooTextFile"')
        lines.append('Object class = "TextGrid"\n')
        lines.append("xmin = 0")
        lines.append(f"xmax = {self.audio_duration}")
        lines.append("tiers? <exists>")
        lines.append(f"size = {len(speakers)}")
        lines.append("item []:")

        # Build tiers
        for tier_idx, speaker in enumerate(speakers, start=1):
            lines.append(f"    item [{tier_idx}]:")
            lines.append('        class = "IntervalTier"')
            lines.append(f'        name = "{speaker}"')
            lines.append("        xmin = 0")
            lines.append(f"        xmax = {self.audio_duration}")

            intervals = []

            for seg in speaker_segments[speaker]:

                # Speech interval
                if self.seperate_words:
                    intervals.append((seg["start"], seg["end"], seg["word"].strip()))
                else:
                    intervals.append((seg["start"], seg["end"], seg["text"].strip()))

            lines.append(f"        intervals: size = {len(intervals)}")

            for i, (start, end, text) in enumerate(intervals, start=1):
                lines.append(f"        intervals [{i}]:")
                lines.append(f"            xmin = {start}")
                lines.append(f"            xmax = {end}")
                lines.append(f'            text = "{text}"')

        # Write file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


# Pipeline
def main():
    M = MEET_Transcription_pipeline(sys.argv[1], sys.argv[2], sys.argv[3], seperate_words = False)
    print("Generating transcript...")
    M.whisper_transcription()

    try:
        job_id = sys.argv[4]
        created_job = False
    except IndexError:
        print("Creating job...")
        job_id = M.create_diarization_job()
        created_job = True

    print("Polling diarization...")
    M.poll_diarization(job_id)

    print("Aligning transcription and diarization...")
    speaker_transcriptions = M.align_outputs()

    if not M.seperate_words:
        speaker_transcriptions= M.stitch_sentences()

    M.write_diarized_textgrid(speaker_transcriptions, f"diarized_precision2_{M.file_name}.TextGrid")
    print("Done!")

    if created_job:
        if os.path.isfile("job_IDs.txt"):
            with open("job_IDs.txt", "a") as f:
                timestamp = time.strftime("%Y-%m-%d %H:%M")
                f.write(f"{M.file_name}: {job_id} - {timestamp}\n")
        else:
            with open("job_IDs.txt", "w") as f:
                timestamp = time.strftime("%Y-%m-%d %H:%M")
                f.write(f"{M.file_name}: {job_id} - {timestamp}\n")

if __name__ =="__main__":
    main()
