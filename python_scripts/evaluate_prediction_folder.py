import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
from src.evaluation.eval_suite import evaluate_folder
from pathlib import Path
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--eval_path", type=Path)
args = parser.parse_args()

eval_path = args.eval_path
vis_path = eval_path / "visualizations"
evaluate_folder(vis_path, eval_path, show_tqdm=True)
