import numpy as np
from navlie.utils import associate_stamps


def test_associate_stamps():
    freq_stamps_1 = 180
    freq_stamps_2 = 50
    t_end = 20

    stamps_1 = list(np.arange(0, t_end, 1 / freq_stamps_1))
    stamps_2 = list(np.arange(0, t_end, 1 / freq_stamps_2))
    matches = associate_stamps(stamps_1, stamps_2)

    assert len(matches) == len(stamps_2)


if __name__ == "__main__":
    test_associate_stamps()
