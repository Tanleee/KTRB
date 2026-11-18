import sys
from pathlib import Path

# Cho phép `pytest` chạy từ thư mục gốc mà không cần cài gói.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
