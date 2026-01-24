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
import argparse
from pathlib import Path
from unittest.mock import MagicMock
from dotenv import load_dotenv

# # Mock missing dependencies
# for mod in ["dtw", "tslearn", "tslearn.clustering", "tensorboard",
#             "tensorboard.backend", "tensorboard.backend.event_processing",
#             "tensorboard.backend.event_processing.event_accumulator"]:
#     sys.modules[mod] = MagicMock()

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
    BASE_DIR = Path(__file__).parent
    RESULT_FILE_NAME = "results.csv"  # Name of the result file

    DESCRIBE_COUNT = 2  # Number of descriptions to generate successfully

    # Load environment variables
    load_dotenv(".env")

    # Add project root to sys.path to import dataloader
    sys.path.append(str(BASE_DIR.parent))
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()


    # 1. Load configuration
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open(BASE_DIR / "instruction.md", "r", encoding="utf-8") as f:
        prompt_template = f.read()

    video_dir = os.path.join(config['data']['vision']['video'])
    result_dir = os.path.join(config['train']['log_dir'])

    os.makedirs(result_dir, exist_ok=True)
    api_key = os.getenv("OPEN_ROUTER_API_KEY")
    if not api_key:
        print("Error: OPEN_ROUTER_API_KEY is missing in .env")
        sys.exit(1)

    again_reader = AgainReader(config=config, again_file_name="clean_data.csv")
    crowd_log_path = os.path.join(config['data']['path'], "crowd_log_data_clean.csv")
    df = pd.read_csv(crowd_log_path)

    # 3. Setup Persistent Results (Robust approach)
    result_file_path = os.path.join(config['data']['log_dir'], RESULT_FILE_NAME)
    results_df = df.copy() # Base is always current input
    for col in ['VideoA_File', 'VideoB_File', 'reason']:
        results_df[col] = None

    # 기존 데이터 로드해서 덮어쓰는 거 같은데, 이럴 필요 없을듯? 더 효율적으로 바꿀 것
    if result_file_path.exists():
        try:
            existing_data = pd.read_csv(result_file_path)
            results_df.update(existing_data) # Merge existing results by index
            print(f"Loaded existing results from {RESULT_FILE_NAME}")
        except Exception as e:
            print(f"Warning: Could not merge existing CSV: {e}")

    # 4. Processing Loop
    success_count = 0

    for idx, row in df.iterrows():
        logs = get_full_logs(row, again_reader)

        video_path = os.path.join(config['data']['vision']['video'], logs['video_file'])
        if not video_path.exists(): video_path = video_path.with_suffix('.mp4')
        
        if not video_path.exists():
            print(f"Skipping index {idx}: File {video_path} not found.")
            continue

        path_a = os.path.join(config['data']['vision']['segments'], f"{idx}_A.mp4")
        path_b = os.path.join(config['data']['vision']['segments'], f"{idx}_B.mp4")

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
                print(f"Done! Result saved for index {idx}.")
            except PermissionError:
                print(f"ERROR: Permission denied. Close '{RESULT_FILE_NAME}' in Excel.")
        else:
            print(f"Failed to get LLM response for index {idx}.")

    print(f"\nProcess finished. Successfully generated {success_count} descriptions.")