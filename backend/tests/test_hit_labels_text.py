"""The text of the marked hits, as "Copia i colpi segnati" gives it."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training" / "hits"))


def test_old_and_new_copies_are_read(tmp_path):
    import train_hits as th

    path = tmp_path / "colpi.txt"
    path.write_text("Set · colpi segnati 4\n"
                    "1:05.86\tTrillo\t2\n"                  # no kind yet
                    "1:02.29 1\n"                           # the oldest: no name
                    "1:07.16\tBeppe\t1\tvolee_rovescio\n"
                    "1:08.00\tUmby 2\t3\tdritto\n", encoding="utf-8")   # a name with a digit
    assert th.read_typed_labels(path) == [(62.29, 1, None), (65.86, 2, None),
                                          (67.16, 1, "volee_rovescio"), (68.0, 3, "dritto")]
    assert th.read_labels(path) == [(62.29, 1), (65.86, 2), (67.16, 1), (68.0, 3)]
    # The labels in the repository still read as before.
    root = Path(__file__).resolve().parents[2]
    assert len(th.read_labels(root / "training/hits/labels/terzo_set.txt")) == 115
    assert len(th.read_labels(root / "training/hits/labels/secondo_set.txt")) == 65
