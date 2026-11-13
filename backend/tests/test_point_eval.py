"""The rallies against the points marked in the video (app.ml.point_eval)."""
from app.ml.point_eval import Point, Span, compare, from_labels, read_points


def test_every_way_a_rally_can_miss_a_point_is_counted_as_such():
    points = [Point(10, 20), Point(40, 50), Point(60, 70), Point(80, 90), Point(100, 110)]
    rallies = [
        Span(10.4, 21.0, serve=True),        # right, 0.4 s late, 1 s long
        Span(40, 44), Span(46, 50),          # one point split in two
        Span(60, 90),                        # two points joined
        # 100-110: lost
        Span(30, 33),                        # between points: false
        Span(200, 210),                      # far outside the marked stretch: not judged
    ]
    report = compare(points, rallies)
    assert report.verdicts == ["giusto", "spezzato", "unito", "unito", "perso"]
    s = report.summary()
    assert (s["points"], s["rallies"], s["one_to_one"]) == (5, 5, 1)
    assert (s["lost"], s["split"], s["joined"], s["false_rallies"]) == (1, 1, 2, 1)
    assert s["serve_seen"] == 1
    assert s["start"]["median_s"] == 0.4 and s["end"]["median_s"] == 1.0
    assert s["start"]["far_off"] == 0


def test_a_few_tenths_of_a_second_apart_still_touch():
    # The marks are made watching the video: a rally ending 0.3 s before
    # the point marked is that point, not a false rally and a lost point.
    report = compare([Point(10, 20)], [Span(5, 9.7)])
    assert report.verdicts == ["giusto"] and report.false_rallies == 0


def test_errors_far_off_are_counted_and_the_90th_percentile_is_absolute():
    points = [Point(10 * k, 10 * k + 5) for k in range(1, 11)]
    rallies = [Span(10 * k + (3.0 if k == 1 else -0.1 * k), 10 * k + 5) for k in range(1, 11)]
    s = compare(points, rallies).summary()
    assert s["one_to_one"] == 10 and s["start"]["far_off"] == 1
    assert s["start"]["p90_abs_s"] == 1.0


def test_no_points_no_judgement():
    assert compare([], [Span(1, 2)]).summary()["rallies"] == 0


def test_the_copied_text_is_read_back():
    text = ("Partita del giovedì · punti segnati 3\n"
            "1:02.30\t1:09.75\tTrillo e Beppe\t2,1\n"
            "1:20.00\t1:31.10\t-\n"
            "0:59.00\t0:58.00\t-\n"                  # ends before it starts: skipped
            "2:00.5\t2:10\tBigna e Umby\t0,3\n")
    assert read_points(text) == [Point(62.3, 69.75, (1, 2)), Point(80.0, 91.1, None),
                                 Point(120.5, 130.0, (0, 3))]


def test_the_stored_labels_are_read_back():
    assert from_labels([[30.0, 40.0, None], [10.0, 20.0, [3, 1]]]) == [Point(10.0, 20.0, (1, 3)),
                                                                       Point(30.0, 40.0, None)]
    assert from_labels(None) == []
