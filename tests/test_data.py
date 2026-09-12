import numpy as np
from tracker.data import index_annotations


def test_index_preserves_original_id_order_and_empty_sparse_frames():
    rows = np.array([[3, 9, 1, 2, 3, 4], [1, 7, 5, 6, 7, 8],
                     [3, 2, 9, 10, 11, 12], [4, 6, 13, 14, 15, 16]])
    result = index_annotations(rows, [1, 2, 3])
    assert list(result) == [1, 2, 3]
    for frame in result:
        selected = rows[rows[:, 0] == frame]
        np.testing.assert_array_equal(result[frame][0], selected[:, 1].astype(np.int64))
        np.testing.assert_array_equal(result[frame][1], selected[:, 2:6].astype(np.float32))
        assert result[frame][0].dtype == np.int64
        assert result[frame][1].dtype == np.float32
    assert result[2][1].shape == (0, 4)


def test_index_accepts_empty_annotations():
    result = index_annotations(np.empty((0, 9)), [1, 2])
    assert all(boxes.shape == (0, 4) and len(ids) == 0 for ids, boxes in result.values())
