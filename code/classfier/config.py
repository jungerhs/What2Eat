"""
数据集构建配置
"""
import os
from pathlib import Path
import dotenv

dotenv.load_dotenv()

# ===== 路径 =====
CLASSFIER_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CLASSFIER_ROOT.parent.parent  # f:/cook项目/cook

DISHES_DIR = PROJECT_ROOT / "data" / "C9" / "dishes"

DATA_DIR = CLASSFIER_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
LOGS_DIR = CLASSFIER_ROOT / "logs"

for _p in (RAW_DIR, PROCESSED_DIR, SYNTHETIC_DIR, LOGS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

REPORTS_DIR = CLASSFIER_ROOT / "reports"
EXPERIMENTS_DIR = REPORTS_DIR / "experiments"

# ===== LR 二分类 (烹饪/非烹饪) =====
LR_DATA_DIR = DATA_DIR / "lr_filter"               # 训练/验证/测试/OOD 合成数据
LR_DATA_DIR.mkdir(parents=True, exist_ok=True)
LR_FILTER_DIR = LR_DATA_DIR                         # 别名 (兼容旧代码)
MODELS_DIR = CLASSFIER_ROOT / "models"
LR_MODEL_DIR = MODELS_DIR / "lr_filter"            # 训练好的模型 (lr.pkl, tfidf.pkl)
LR_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ===== 跳过目录 =====
SKIP_DIRS = {"HowToCook-master", ".git", "__pycache__"}

# ===== 4 类标签 =====
LABELS = ["general", "detail", "multi-hop", "recommend"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for i, label in enumerate(LABELS)}

# ===== 目录名 → 中文分类 =====
DIR_TO_CATEGORY = {
    "aquatic": "水产",
    "breakfast": "早餐",
    "condiment": "调料",
    "dessert": "甜品",
    "drink": "饮料",
    "meat_dish": "荤菜",
    "semi-finished": "半成品",
    "soup": "汤类",
    "staple": "主食",
    "vegetable_dish": "素菜",
}

# ===== 场景词 (multi-hop / recommend 模板用) =====
SCENES = [
    "早餐", "午餐", "晚餐", "夜宵", "宵夜", "聚会", "招待客人",
    "一个人吃", "两人份", "夏天", "冬天", "春天", "秋天",
    "生日", "节日", "减脂", "增肌", "新手", "懒人",
    "快手菜", "下饭", "下酒", "便当", "带饭", "懒人晚餐",
    "待客", "宴客", "家宴",
]

# ===== 数字/比例疑问词 (detail 模板判别用) =====
DETAIL_HINT_RE = [
    r"\d+", r"多少", r"多久", r"几勺", r"几片", r"几分钟", r"几度",
    r"比例", r"火候", r"温度", r"第几[步个]", r"什么时候", r"哪个[时候步]",
    r"放什么", r"加什么", r"用什么[来做替代]",
]

# ===== LLM (Kimi / Moonshot, OpenAI 兼容) =====
# 优先用项目里 .env 的变量名 (API_KEY / base_url), 兼容 MOONSHOT_API_KEY
LLM_API_KEY = os.getenv("API_KEY") or os.getenv("MOONSHOT_API_KEY")
LLM_BASE_URL = os.getenv("BASE_URL") or os.getenv("MOONSHOT_BASE_URL") or "https://api.moonshot.cn/v1"
LLM_MODEL = os.getenv("LLM_MODEL") or os.getenv("MOONSHOT_MODEL") or "kimi-k2-0711-preview"

# ===== LLM Provider 切换 (online / ollama) =====
# LLM_PROVIDER=ollama 时，所有 LLM 调用走本地 Ollama (OpenAI 兼容端点)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "online").lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")  # Ollama 接受任意 key

if LLM_PROVIDER == "ollama":
    LLM_API_KEY = OLLAMA_API_KEY
    LLM_BASE_URL = OLLAMA_BASE_URL
    LLM_MODEL = OLLAMA_MODEL

# ===== 数据生成目标 =====
TARGET_PER_CLASS_TRAIN = 2000   # 每类训练目标
TARGET_PER_CLASS_TEST = 250     # 每类测试目标
TEMPLATE_PER_DISH = 1           # 每个菜品每类生成几条
LLM_PER_CLASS = 2000            # LLM 每类合成目标
LLM_BATCH_SIZE = 50             # 一批调用条数
LLM_CONCURRENCY = 8            # 并发请求数
LLM_TEMPERATURE = 0.8           # 合成数据需要随机性

# ===== 切分比例 =====
TRAIN_RATIO = 0.8
DEV_RATIO = 0.1
TEST_RATIO = 0.1

# ===== 长度过滤 =====
MIN_LEN = 5
MAX_LEN = 60

# ===== 数据增强：cooking 训练样本尾部补问号比例 =====
# 真实用户输入中带问号的占比约 35-45%，与 OOD 合成数据 (~37%) 持平；
# 若 cooking 侧 0% 带问号，LR 会把 "？" 学成强 OOD 信号
# → 烹饪 query 一加 "？" 就被 reject。
QM_RATIO = 0.4

# ===== 二次打标一致率阈值 =====
DOUBLE_CHECK_AGREEMENT_THRESHOLD = 0.85
DISPUTE_RATIO = 0.05  # 抽 5% 做二次打标

# ===== 去重 =====
SIMHASH_HAMMING_THRESHOLD = 5

# ===== LR 分类器训练 / 推理 =====
LR_TARGET_NEG = 500              # 目标 OOD 负样本数 (LLM 合成) (实验扩量)
LR_TFIDF_MAX_FEATURES = 20000
LR_TFIDF_NGRAM = (1, 2)
LR_TFIDF_MIN_DF = 2
LR_C = 1.0
LR_MAX_ITER = 1000

# ===== Pipeline 阈值 (predictor.py) =====
LR_THRESHOLD_LOW = 0.5           # LR 极不自信 → 直接 unknown
LR_THRESHOLD_HIGH = 0.6          # LR 中间地带 → 调 LLM is_cooking 兜底
BERT_THRESHOLD = 0.6             # BERT 不自信 → 调 LLM 4 类兜底
USE_LLM_FALLBACK = True          # 是否开 LLM 兑底

# ===== 日志 =====
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
