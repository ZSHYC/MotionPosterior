import numpy as np

from core.windowing import iter_sliding_windows


class _Frames:
    def __init__(self, count):
        self.count = count
        self.index = 0

    def read(self):
        if self.index >= self.count:
            return False, None
        value = self.index
        self.index += 1
        return True, np.array([[value]], dtype=np.uint8)


def test_causal_windows_never_use_future_output_frame():
    windows = list(iter_sliding_windows(_Frames(7), 5, 'causal'))
    assert [frame_index for _, _, frame_index in windows] == list(range(7))
    assert [int(frames[position][0, 0]) for frames, position, _ in windows] == list(range(7))


def test_center_windows_emit_center_frame_for_each_index():
    windows = list(iter_sliding_windows(_Frames(7), 5, 'center'))
    assert [frame_index for _, _, frame_index in windows] == list(range(7))
    assert [int(frames[position][0, 0]) for frames, position, _ in windows] == list(range(7))
