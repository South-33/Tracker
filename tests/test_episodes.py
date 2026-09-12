import numpy as np

from tracker.episodes import (
    discover_gap_episodes,
    episode_is_evaluable,
    score_episode,
    summarize_episodes,
)


EVENT = {"person_id": 1, "pre_start": 1, "gap_start": 3, "return_frame": 5,
         "deadline_seconds": .5, "followup_seconds": 2., "kind": "occlusion"}


def associations():
    return {frame: ({1: 10, 2: 20} if frame < 3 or frame >= 5 else {2: 20}) for frame in range(1, 11)}


def test_wrong_person_reuse_wins_even_after_correct_return():
    data = associations()
    data[8] = {1: 30, 2: 10}
    result = score_episode(EVENT, data, 2)
    assert result["outcome"] == "wrong_person_reuse"
    assert result["wrong_person_id"] == 2 and result["wrong_person_frame"] == 8


def test_delayed_switch_is_not_success():
    data = associations()
    data[8] = {1: 30, 2: 20}
    result = score_episode(EVENT, data, 2)
    assert result["outcome"] == "false_new" and result["delayed_switch"]


def test_pre_gap_failure_is_kept_in_overall_denominator():
    good = score_episode(EVENT, associations(), 2)
    data = associations()
    data[1] = data[2] = {}
    failed = score_episode(EVENT, data, 2)
    summary = summarize_episodes([good, failed])
    assert summary["unconditional_recovery"] == .5
    assert summary["conditional_recovery"] == 1


def test_abstention_and_missed_observation_are_distinct():
    data = associations()
    data[5] = data[6] = {1: None}
    assert score_episode(EVENT, data, 2)["outcome"] == "abstention"
    data[5] = data[6] = {}
    assert score_episode(EVENT, data, 2)["outcome"] == "lost_observation"


def test_gap_discovery_keeps_only_scoreable_same_identity_returns():
    truth = {}
    for frame in range(1, 21):
        ids = [2]
        if frame <= 5 or frame >= 10:
            ids.append(1)
        truth[frame] = (np.asarray(ids), np.empty((len(ids), 4)))
    events = discover_gap_episodes(truth, fps=2, min_gap_seconds=1, pre_seconds=1,
                                   deadline_seconds=.5, followup_seconds=1)
    assert len(events) == 1
    event = events[0]
    assert event["person_id"] == 1
    assert event["gap_start"] == 6 and event["return_frame"] == 10
    assert event["gap_frames"] == 4 and event["gap_seconds"] == 2


def test_episode_evaluable_requires_the_complete_scoring_window():
    assert episode_is_evaluable(EVENT, associations(), 2)
    partial = {frame: value for frame, value in associations().items() if frame <= 6}
    assert not episode_is_evaluable(EVENT, partial, 2)
