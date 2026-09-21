# import official package
import os
import json
import numpy as np
import pickle   # Used to import the function for reading .dat files from the .dat library.
from scipy.io import loadmat    # Used to import the function for reading MATLAB .mat files from the SciPy library.

# import person package && config
from src.EEG.api.Confings.Config import Config

cache_path = Config.cache_path      # cache path
cache_meta_path = str(cache_path) + ".meta.json"
deap_data_path = Config.deap_data_path      # deap data path
eav_data_path = Config.eav_data_path    # eav data path

CACHE_VERSION = Config.cache_version

# 两个数据集的固定结构，_build() 和 _expected_total() 共用，避免两边写死不同的数。
DEAP_TRIALS = 40        # DEAP 每个受试者 40 个试次
DEAP_SEGS = 4           # 一个试次切成 4 段
EAV_SEGS = 200          # EAV 每个受试者 200 个 segment
EAV_CHUNKS = 5          # 一个 segment 切成 5 块


# EegPlayers
class EegPlayer:
    __slots__ = ('data', 'valence', 'subject', 'trial', 'source')

    def __init__(self, data, valence, subject, trial, source):
        self.data = data
        self.valence = valence
        self.subject = subject      # 受试者 id（deap: 's01'，eav: 'subject1'）
        self.trial = trial          # 试次 id：DEAP 是试次号，EAV 是 segment 号
        self.source = source        # 'deap' | 'eav'


# DataRead
_play_list = None


def data_read() -> np.ndarray:
    """返回全部样本。

    结果在进程内只加载一次：TrainDataSet / KnownDataSet / UnknownDataSet 都要用
    同一份数据切分，之前每个 Dataset 各调一次 data_read()，等于把 11GB 的缓存
    在内存里反序列化了三遍。
    """
    global _play_list
    if _play_list is None:
        _play_list = _load_cache()
        if _play_list is None:
            _play_list = _build()
            _save_cache(_play_list)
    return _play_list


def _expected_total() -> int:
    """按源数据的文件数推算应有的样本总量，用来识别缓存过期。"""
    n_deap = len([d for d in os.listdir(deap_data_path) if d.endswith(".dat")])
    n_eav = len([d for d in os.listdir(eav_data_path)
                 if d != "GitHub_Codes" and os.path.isdir(os.path.join(eav_data_path, d))])
    return n_deap * DEAP_TRIALS * DEAP_SEGS + n_eav * EAV_SEGS * EAV_CHUNKS


def _load_cache():
    """校验并加载缓存；任何一项对不上就返回 None，交给 _build() 重建。

    以前只判断文件是否存在就直接 np.load，改了 data_read 的读取逻辑或换了切分
    方式之后，旧缓存会被继续使用而没有任何提示 —— 这会让"代码改了但结果没变"
    变得极难排查。
    """
    if not (os.path.exists(cache_path) and os.path.exists(cache_meta_path)):
        return None

    try:
        with open(cache_meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError):
        print("缓存元数据损坏，将重建缓存")
        return None

    if meta.get("version") != CACHE_VERSION:
        print(f"缓存版本不匹配（磁盘 {meta.get('version')} != 代码 {CACHE_VERSION}），将重建缓存")
        return None

    n_meta = meta.get("n")

    try:
        expected = _expected_total()
    except OSError:
        # 源数据目录不可读（比如换了机器、只带了缓存）：跳过数量校验，只信版本号
        print("源数据目录不可读，跳过缓存数量校验")
        expected = None

    if expected is not None and n_meta != expected:
        print(f"缓存样本数与源数据不符（缓存 {n_meta} != 预期 {expected}），将重建缓存")
        return None

    play_list = np.load(cache_path, allow_pickle=True)

    if n_meta != len(play_list):
        print(f"缓存文件实际长度 {len(play_list)} != 记录值 {n_meta}，将重建缓存")
        return None

    print(f"read cache data ({len(play_list)} samples)")
    return play_list


def _build() -> np.ndarray:
    play_list = []

    # ---------------- deap data read ----------------
    for dirs in sorted(os.listdir(deap_data_path)):
        if not dirs.endswith(".dat"):
            continue

        subject_id = os.path.splitext(dirs)[0]

        with open(os.path.join(deap_data_path, dirs), "rb") as f:
            subject = pickle.load(f, encoding="latin1")

        label = subject['labels']
        # 前 32 个电极为脑电数据，这里取前 30 个以对齐 EAV 的 30 通道
        data = np.array(subject['data'][:, :30], dtype=np.float32)

        for i in range(DEAP_TRIALS):
            row = label[i].tolist()

            # DEAP 的效价是 1-9 的浮点评分，中点是 5。
            # 原来写 int(row[0]) <= 5，int() 是截断，5.9 也会被判成 0（消极），
            # 等于把阈值悄悄挪到了 6。这里直接按浮点比。
            valence = 0 if row[0] <= 5 else 1

            for j in range(DEAP_SEGS):
                # 8064 个采样点切成 4 段互不重叠的 2000 点。
                # 之前写成 j:j+2016，4 段只平移 1 个点，几乎完全相同，
                # 等于把每个试次复制了 4 份，训练精度全靠背数据。
                play_list.append(EegPlayer(
                    data[i][:, j * 2000:(j + 1) * 2000].transpose(-1, 0).copy(),
                    valence, subject_id, i, "deap"))

    # ---------------- eav data read ----------------
    # 按目录逐个配对 _eeg.mat / _eeg_label.mat。
    # 原来是把所有文件拍平成一个列表再取 [0::2] / [1::2]，依赖 os.listdir 的
    # 返回顺序刚好是"数据文件在前、标签文件在后"。任何目录多出一个文件，
    # 后面所有受试者的标签就会整体错位一格。
    pairs = []
    for dirs in sorted(os.listdir(eav_data_path)):
        if dirs == "GitHub_Codes":
            continue

        eeg_dir = os.path.join(eav_data_path, dirs, 'EEG')
        if not os.path.isdir(eeg_dir):
            continue

        files = sorted(os.listdir(eeg_dir))
        eeg_files = [f for f in files if f.endswith('_eeg.mat')]
        label_files = [f for f in files if f.endswith('_eeg_label.mat')]

        if len(eeg_files) != 1 or len(label_files) != 1:
            raise RuntimeError(
                f"{eeg_dir} 期望恰好 1 个 _eeg.mat 和 1 个 _eeg_label.mat，实际为 {files}")

        pairs.append((dirs,
                      os.path.join(eeg_dir, eeg_files[0]),
                      os.path.join(eeg_dir, label_files[0])))

    for i, (subject_id, eeg_path, label_path) in enumerate(pairs):
        print("数据处理", i + 1, "/", len(pairs))
        print("数据提取")

        # 部分文件的变量名是 seg,部分是 seg1
        valance = loadmat(eeg_path)
        valance = valance['seg'] if 'seg' in valance else valance['seg1']
        valance = valance.transpose(-1, 0, 1)   # (10000, 30, 200) -> (200, 10000, 30)

        label = loadmat(label_path)
        label = label['label'].transpose(-1, 0)     # (10, 200) -> (200, 10)

        print("数据提取完毕")

        for s in range(valance.shape[0]):       # 200 个 segment
            valance_data = np.array(valance[s], dtype=np.float32)   # (10000, 30)

            label_data = get_negative_label(label[s])
            for j in range(EAV_CHUNKS):         # 一段数据切成五分
                play_list.append(EegPlayer(
                    valance_data[2000 * j: 2000 * (j + 1)].copy(),
                    label_data, subject_id, s, "eav"))

    return np.stack(play_list)


def _save_cache(play_list) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, play_list, allow_pickle=True)
    with open(cache_meta_path, "w", encoding="utf-8") as f:
        json.dump({"version": CACHE_VERSION, "n": len(play_list)}, f,
                  ensure_ascii=False, indent=2)
    print(f"cache saved: {cache_path} ({len(play_list)} samples)")


# functions
# EAV 官方 10 类索引，每一对是同一情绪的"听 / 说"两个任务：
#     0, 1 = neutral    中性
#     2, 3 = sadness    悲伤
#     4, 5 = anger      愤怒
#     6, 7 = happiness  高兴
#     8, 9 = calmness   平静
#
# 依据是数据集自带的参考实现
# eav_dataset/GitHub_Codes/emotionCNN_segmented_all_sub_5class.py 顶部的
# TRIGGER_START_TRIAL_* 触发表（NEU=0, S=2, A=4, H=6, R=8），
# 以及 EAV 论文（Scientific Data 2024, doi:10.1038/s41597-024-03838-4）给出的
# 5 类情绪 neutral / anger / happiness / sadness / calmness。
#
# 旧版本的 docstring 把 2,3 写成"愤怒"、6,7 写成"悲伤"，两个都反了。
# 结果是 {2,3,6,7} 实际取到的是 {悲伤, 高兴} —— 把"高兴"标成了消极，
# 把"愤怒"标成了积极。这不是随机噪声而是整组系统性反转，占 EAV 的 2/5。
# 消极面应当是 悲伤 + 愤怒。
_NEGATIVE_CLASSES = (2, 3, 4, 5)


def get_negative_label(label) -> int:
    """
    0 = 消极（悲伤 / 愤怒）
    1 = 积极（中性 / 高兴 / 平静）
    """
    cls = int(np.argmax(label, axis=-1))
    return 0 if cls in _NEGATIVE_CLASSES else 1


if __name__ == "__main__":
    play_list = data_read()

    print(f"total: {len(play_list)}")
