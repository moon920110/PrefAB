# llm/description_unlabeled.py
import os
import sys
import json
import yaml
import time
import base64
import subprocess
import tempfile
import argparse
import logging
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# --- Path Setup ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- Custom Imports ---
from dataloader.again_reader import AgainReader
from llm.llm_client import get_appraisal_analysis
from llm.prompt.task_prompt import TASK_INSTRUCTION_UNLABELED
from llm.prompt.game_prompt import PROMPT_MAP

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Constants ---
FEATURE_MAP = {
    'novelty': 'Novelty',
    'goal_relevance': 'GoalRelevance',
    'outcome_probability': 'OutcomeProbability',
    'goal_conduciveness': 'GoalConduciveness',
    'urgency': 'Urgency',
    'coping_potential': 'CopingPotential'
}


# --- Core Logic Functions ---

def load_config(config_rel_path):
    """Loads configuration yaml relative to project root."""
    config_path = PROJECT_ROOT / config_rel_path
    if not config_path.exists():
        logger.critical(f"Config not found: {config_path}")
        sys.exit(1)
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_datasets(config):
    """Initializes the AgainReader and loads the task list (crowd log)."""
    try:
        reader = AgainReader(config=config, again_file_name="clean_data.csv")

        crowd_log_path = Path(config['data']['path']) / "crowd_log_data_clean.csv"
        if not crowd_log_path.exists():
            raise FileNotFoundError(f"Crowd log not found: {crowd_log_path}")

        df = pd.read_csv(crowd_log_path)
        return reader, df
    except Exception as e:
        logger.critical(f"Data loading failed: {e}")
        sys.exit(1)


def get_resume_state(result_path, source_df):
    """Handles logic to resume from existing results file."""
    if not result_path.exists():
        df = source_df.copy()
        df['analysis_json'] = None
        df['valid_features'] = None
        df['consistency_score'] = 0
        return df, set()

    try:
        existing_df = pd.read_csv(result_path)
        processed_indices = set(existing_df[existing_df['analysis_json'].notna()].index)

        results_df = existing_df.copy()
        if len(results_df) != len(source_df):
            results_df = source_df.copy()
            results_df.update(existing_df)

        return results_df, processed_indices
    except Exception as e:
        logger.warning(f"Resume failed ({e}). Starting fresh.")
        df = source_df.copy()
        df['analysis_json'] = None
        df['valid_features'] = None
        df['consistency_score'] = 0
        return df, set()


def get_game_logs(row, reader, duration=3):
    """Extracts session data and specific time window logs."""
    try:
        video_file = row['VideoFile']
        session_id = video_file.split('_')[-1].split('.')[0]
        game_name = row['GameName']

        game_data = reader.game_info_by_name(game_name)
        if game_data is None or game_data.empty: return None

        session_data = game_data[game_data['session_id'] == session_id]
        if session_data.empty: return None

        return {
            'game_name': game_name,
            'video_file': video_file,
            'start_a': float(row['StartTimeA']),
            'start_b': float(row['StartTimeB']),
            'labels': {k: row.get(v) for k, v in FEATURE_MAP.items()}
        }
    except Exception as e:
        logger.error(f"Log extraction error: {e}")
        return None


def verify_consistency(llm_result, human_labels):
    """Checks if LLM predictions match human labels."""
    valid_features = []
    matches = 0

    for json_key, _ in FEATURE_MAP.items():
        if json_key not in llm_result: continue

        llm_pred = llm_result[json_key].get('predicted_winner')
        human_label = human_labels.get(json_key)

        if llm_pred and human_label and llm_pred == human_label:
            matches += 1
            valid_features.append(json_key)

    return valid_features, matches


def encode_video(video_path, start, duration=3):
    """Encodes a video segment to base64 using ffmpeg."""
    tmp_path = Path(tempfile.gettempdir()) / f"tmp_{time.time()}_{os.getpid()}.mp4"
    video_path_str = str(video_path)
    tmp_path_str = str(tmp_path)

    opts = ['-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28', '-vf', 'scale=640:-2', '-an']

    try:
        # fast seek -> slow seek
        for seek_opt in [['-ss', str(start), '-i', video_path_str], ['-i', video_path_str, '-ss', str(start)]]:
            cmd = ['ffmpeg', '-y'] + seek_opt + ['-t', str(duration)] + opts + [tmp_path_str]
            subprocess.run(cmd, capture_output=True)

            if tmp_path.exists() and tmp_path.stat().st_size > 0:
                with open(tmp_path, "rb") as f:
                    return base64.b64encode(f.read()).decode('utf-8')

    except Exception as e:
        logger.error(f"Encoding error: {e}")
    finally:
        if tmp_path.exists():
            try:
                os.remove(tmp_path)
            except:
                pass
    return None


def process_single_task(idx, row, reader, video_root, api_key):
    """Executes the pipeline for a single row: Log -> Video -> LLM -> Check."""

    # 1. Get Logs
    logs = get_game_logs(row, reader)
    if not logs: return None

    # 2. Check Video File
    video_path = video_root / logs['video_file']
    if not video_path.exists():
        video_path = video_path.with_suffix('.mp4')
    if not video_path.exists():
        if idx % 10 == 0: logger.warning(f"Video missing: {video_path}")
        return None

    # 3. Encode Segments
    b64_a = encode_video(video_path, logs['start_a'])
    b64_b = encode_video(video_path, logs['start_b'])
    if not (b64_a and b64_b): return None

    # 4. Construct Prompt
    context = PROMPT_MAP.get(logs['game_name'], "No context available.")
    system_prompt = f"{TASK_INSTRUCTION_UNLABELED}\n\nGame Information:\n{context}"

    # 5. Call LLM
    result = get_appraisal_analysis(api_key, system_prompt, "Analyze these videos.", b64_a, b64_b)

    if result:
        valid_feats, score = verify_consistency(result, logs['labels'])
        return {
            'analysis_json': json.dumps(result, ensure_ascii=False),
            'valid_features': json.dumps(valid_feats),
            'consistency_score': score
        }
    return None


# --- Main Execution ---

def main():
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()

    # 1. Setup
    config = load_config(args.config)
    api_key = os.getenv("OPEN_ROUTER_API_KEY")
    if not api_key: sys.exit("Missing API Key")

    video_dir = Path(config['data']['vision']['video'])
    log_dir = Path(config['data']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    result_path = log_dir / "results.csv"

    # 2. Data
    reader, source_df = load_datasets(config)
    results_df, processed_indices = get_resume_state(result_path, source_df)

    logger.info(f"Starting processing. {len(processed_indices)} items already done.")

    # 3. Loop
    success_count = 0
    for idx, row in source_df.iterrows():
        # Skip processed
        if idx in processed_indices and pd.notna(results_df.loc[idx, 'analysis_json']):
            continue

        # Process Task
        output = process_single_task(idx, row, reader, video_dir, api_key)

        if output:
            results_df.loc[idx, ['analysis_json', 'valid_features', 'consistency_score']] = \
                [output['analysis_json'], output['valid_features'], output['consistency_score']]

            try:
                results_df.to_csv(result_path, index=False, encoding='utf-8-sig')
                success_count += 1
                logger.info(f"[{idx}] Success. Match: {output['consistency_score']}/6. Valid: {output['valid_features']}")
            except PermissionError:
                logger.error("Permission denied saving CSV.")

            time.sleep(1)
        else:
            logger.error(f"[{idx}] Failed or Skipped.")

    logger.info(f"Finished. Processed {success_count} new items.")


if __name__ == "__main__":
    main()