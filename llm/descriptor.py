import os
import requests
import json
import yaml
import sys
import pandas as pd
import base64
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock
from dotenv import load_dotenv

# --- Constants & Configuration ---
BASE_DIR = Path(__file__).parent
CONFIG_FILE_PATH = BASE_DIR / "../config" / "config.yaml"                 # Config for AgainReader
VIDEO_FOLDER_PATH = r"D:\Projects\PREFAB\Data\videos_mp4"   # Path to video files
RESULT_FOLDER_PATH = r"D:\Projects\PREFAB\Data\results"     # Path to save results
RESULT_FILE_NAME = "results.csv"                            # Name of the result file
INDEX_OFFSET_FOR_PANDAS = 0 # Offset to match pandas row numbers (0-indexed)
INDEX_OFFSET_FOR_EXCEL = 2  # Offset to match Excel row numbers (1-indexed + header)
INDEX_OFFSET = INDEX_OFFSET_FOR_EXCEL # If you don't want to use Excel row numbers, set this to 0                                            

START_EXCEL_INDEX = 10  # Row number in Excel to start from
DESCRIBE_COUNT = 2      # Number of descriptions to generate successfully

# Load environment variables
load_dotenv(BASE_DIR / ".env")

# Add project root to sys.path to import dataloader
sys.path.append(str(BASE_DIR.parent))

# Mock missing dependencies
for mod in ["dtw", "tslearn", "tslearn.clustering", "tensorboard", 
            "tensorboard.backend", "tensorboard.backend.event_processing", 
            "tensorboard.backend.event_processing.event_accumulator"]:
    sys.modules[mod] = MagicMock()

from dataloader.again_reader import AgainReader

# --- Core Functions ---

def get_full_logs(row, reader: AgainReader, duration=3):
    """Extracts relevant gameplay logs for specified time windows."""
    try:
        video_file = row['VideoFile']
        session_id = video_file.split('_')[-1].split('.')[0]
        game_name = row['GameName']
        
        game_data = reader.game_info_by_name(game_name)
        if game_data is None or game_data.empty:
            return None
            
        session_data = game_data[game_data['session_id'] == session_id].copy()
        if session_data.empty:
            return None

        session_data['time_sec'] = pd.to_timedelta(session_data['time_index']).dt.total_seconds()
        
        start_a = float(row['StartTimeA'])
        start_b = float(row['StartTimeB'])
        
        log_a = session_data[(session_data['time_sec'] >= start_a) & (session_data['time_sec'] <= start_a + duration)]
        log_b = session_data[(session_data['time_sec'] >= start_b) & (session_data['time_sec'] <= start_b + duration)]
        
        return {
            'participant_id': row['ParticipantID'],
            'game': game_name,
            'session': session_id,
            'video_file': video_file,
            'start_a': start_a, 'log_a': log_a,
            'start_b': start_b, 'log_b': log_b,
            'appraisal': {
                'novelty': row.get('Novelty'),
                'goal_relevance': row.get('GoalRelevance'),
                'outcome_probability': row.get('OutcomeProbability'),
                'goal_conduciveness': row.get('GoalConduciveness'),
                'urgency': row.get('Urgency'),
                'coping_potential': row.get('CopingPotential')
            }
        }
    except Exception as e:
        print(f"Error extracting logs: {e}")
        return None

def encode_video_segment(video_path, start_time, duration=3, output_save_path=None):
    """Trims video using ffmpeg. Tries fast seeking first, then slow seeking."""
    tmp_path = output_save_path or Path(tempfile.gettempdir()) / f"tmp_{time.time()}.mp4"
    
    try:
        # Strategy 1: Fast Seek (-ss before -i)
        cmd = [
            'ffmpeg', '-y', '-ss', str(start_time), '-i', str(video_path),
            '-t', str(duration), '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28', 
            '-vf', 'scale=640:-2', '-an', str(tmp_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            # Strategy 2: Slow Seek (-ss after -i) for robustness
            cmd_slow = [
                'ffmpeg', '-y', '-i', str(video_path), '-ss', str(start_time),
                '-t', str(duration), '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28', 
                '-vf', 'scale=640:-2', '-an', str(tmp_path)
            ]
            result = subprocess.run(cmd_slow, capture_output=True, text=True)
            
        if result.returncode == 0 and os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
            with open(tmp_path, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        else:
            print(f"Video encoding failed at {start_time}s. FFmpeg Output: {result.stderr}")
    except Exception as e:
        print(f"Unexpected error during video encoding: {e}")
    finally:
        if not output_save_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
    return None

def call_openrouter_api(api_key, prompt, base64_a, base64_b):
    """Performs the LLM request via OpenRouter."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "google/gemini-2.5-flash",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "text", "text": "Here is Video A:"},
                {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{base64_a}"}},
                {"type": "text", "text": "And here is Video B:"},
                {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{base64_b}"}}
            ]
        }]
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"API Request Error: {e}")
        return None

# --- Main Logic ---

if __name__ == "__main__":
    os.makedirs(RESULT_FOLDER_PATH, exist_ok=True)
    api_key = os.getenv("OPEN_ROUTER_API_KEY")
    if not api_key:
        print("Error: OPEN_ROUTER_API_KEY is missing in .env")
        sys.exit(1)

    # 1. Load configuration
    with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    with open(BASE_DIR / "instruction.md", "r", encoding="utf-8") as f:
        prompt_template = f.read()

    # 2. Setup Data Readers
    original_read_csv = pd.read_csv
    pd.read_csv = lambda *args, **kwargs: (
        pd.DataFrame(columns=['ParticipantID', 'Gender', 'Play Frequency', 'Gamer', 'Genre'])
        if 'biographical_data_with_genre.csv' in str(args[0])
        else original_read_csv(*args, **kwargs)
    )
    
    try:
        again_reader = AgainReader(config=config, again_file_name="clean_data.csv")
        crowd_log_path = Path(config["data"]["path"]) / "crowd_log_data_clean.csv"
        df = pd.read_csv(crowd_log_path, sep='\t')
        if len(df.columns) < 2: df = pd.read_csv(crowd_log_path)
    finally:
        pd.read_csv = original_read_csv

    # 3. Setup Persistent Results (Robust approach)
    result_file_path = Path(RESULT_FOLDER_PATH) / RESULT_FILE_NAME
    results_df = df.copy() # Base is always current input
    for col in ['VideoA_File', 'VideoB_File', 'reason']:
        results_df[col] = None
        
    if result_file_path.exists():
        try:
            existing_data = pd.read_csv(result_file_path)
            results_df.update(existing_data) # Merge existing results by index
            print(f"Loaded existing results from {RESULT_FILE_NAME}")
        except Exception as e:
            print(f"Warning: Could not merge existing CSV: {e}")

    # 4. Processing Loop
    success_count = 0
    start_index = START_EXCEL_INDEX - INDEX_OFFSET
    
    print(f"Starting processing from index {start_index} (Excel row {START_EXCEL_INDEX})")

    for idx, row in df.iterrows():
        if idx < start_index: continue
        if idx - start_index >= DESCRIBE_COUNT: break

        print(f"\nProcessing index {idx} (excel: {idx + INDEX_OFFSET_FOR_EXCEL})...")
        
        # Get logs
        logs = get_full_logs(row, again_reader)
        if not logs:
            print(f"Skipping index {idx} (excel: {idx + INDEX_OFFSET_FOR_EXCEL}): No matching session logs.")
            continue

        video_path = Path(VIDEO_FOLDER_PATH) / logs['video_file']
        if not video_path.exists(): video_path = video_path.with_suffix('.mp4')
        
        if not video_path.exists():
            print(f"Skipping index {idx} (excel: {idx + INDEX_OFFSET_FOR_EXCEL}): File {video_path} not found.")
            continue

        save_idx = idx + INDEX_OFFSET
        path_a = Path(RESULT_FOLDER_PATH) / f"{save_idx}_A.mp4"
        path_b = Path(RESULT_FOLDER_PATH) / f"{save_idx}_B.mp4"

        b64_a = encode_video_segment(video_path, logs['start_a'], output_save_path=path_a)
        b64_b = encode_video_segment(video_path, logs['start_b'], output_save_path=path_b)

        if not (b64_a and b64_b):
            continue

        # Get LLM response
        prompt = prompt_template.format(**logs['appraisal'])
        reason = call_openrouter_api(api_key, prompt, b64_a, b64_b)

        if reason:
            results_df.loc[idx, ['VideoA_File', 'VideoB_File', 'reason']] = [path_a.name, path_b.name, reason]
            try:
                results_df.to_csv(result_file_path, index=False, encoding='utf-8-sig')
                success_count += 1
                print(f"Done! Result saved for index {idx} (excel: {idx + INDEX_OFFSET_FOR_EXCEL}).")
            except PermissionError:
                print(f"ERROR: Permission denied. Close '{RESULT_FILE_NAME}' in Excel.")
        else:
            print(f"Failed to get LLM response for index {idx} (excel: {idx + INDEX_OFFSET_FOR_EXCEL}).")

    print(f"\nProcess finished. Successfully generated {success_count} descriptions.")