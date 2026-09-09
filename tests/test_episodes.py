from tracker.episodes import score_episode, summarize_episodes


EVENT = {"person_id": 1, "pre_start": 1, "gap_start": 3, "return_frame": 5,
         "deadline_seconds": .5, "followup_seconds": 2., "kind": "occlusion"}


def associations():
    return {frame: ({1: 10, 2: 20} if frame < 3 or frame >= 5 else {2: 20}) for frame in range(1, 11)}


def test_wrong_person_reuse_wins_even_after_correct_return():
    data = associations()
    data[8] = {1: 30, 2: 10}
    assert score_episode(EVENT, data, 2)["outcome"] == "wrong_person_reuse"


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
