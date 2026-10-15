"""CurriculumCallback._collect 单元测试（方案 A1：课程按回合计数 + 结局统计）。
运行：source .venv-fr3/bin/activate && pytest rl/test_train_ppo.py -v -p no:cacheprovider
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from train_ppo import CurriculumCallback


def make_callback():
    return CurriculumCallback([('A', {}), ('B', {})], window=10, promote_at=0.9, verbose=0)


def test_collect_ignores_running_steps():
    """回合未结束（done=False）绝不计数：旧 bug 是按步计数导致
    success_rate 上限 ≈ 1/回合长度，成功率 100% 也只有 ~1.4%。"""
    cb = make_callback()
    cb._collect([{'is_success': True, 'status': 'success'}] * 5, [False] * 5)
    assert len(cb.window) == 0
    assert sum(cb.status_counts.values()) == 0


def test_collect_counts_episode_ends():
    """回合结束才计数：成功窗口与结局统计都按回合记录。"""
    cb = make_callback()
    cb._collect(
        [{'is_success': True, 'status': 'success'},
         {'is_success': False, 'status': 'timeout'},
         {'is_success': False, 'status': 'out_of_bounds'}],
        [True, True, True])
    assert list(cb.window) == [True, False, False]
    assert cb.status_counts == {'success': 1, 'timeout': 1, 'out_of_bounds': 1}


def test_collect_status_missing_or_none():
    """status 为 None 归入 unknown（如 reset 后立即结束等异常情形）；is_success 缺失按 False 计。"""
    cb = make_callback()
    cb._collect([{'status': None}, {'is_success': True, 'status': None}],
                [True, True])
    assert list(cb.window) == [False, True]
    assert cb.status_counts == {'unknown': 2}
