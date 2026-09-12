"""Score explicitly defined disappearance episodes from evaluation associations.

Each frame maps ground-truth IDs to a public ID, None for observed-but-abstained,
or has no entry for a missed observation. Association is evaluation-only.
"""
from collections import Counter


def episode_frame_range(event, fps):
    """Return the inclusive frame range required to score one episode."""
    return_frame = event["return_frame"]
    deadline = return_frame + round(event.get("deadline_seconds", .5) * fps)
    end = deadline + round(event.get("followup_seconds", 2.) * fps)
    return range(event["pre_start"], end + 1)


def episode_is_evaluable(event, associations, fps):
    """Whether the full episode window exists in one uninterrupted rollout."""
    return all(frame in associations for frame in episode_frame_range(event, fps))


def discover_gap_episodes(truth, fps, min_gap_seconds=.5, max_gap_seconds=10.,
                          pre_seconds=.5, deadline_seconds=.5, followup_seconds=2.):
    """Find same-ID annotation gaps that can be scored without resetting state.

    These are candidate disappearance/re-entry episodes, not claims about why the
    person is absent. Visual review is still required to label exit vs occlusion.
    """
    frames = sorted(truth)
    if not frames or frames != list(range(frames[0], frames[-1] + 1)):
        raise ValueError("Episode discovery requires contiguous annotations")
    present = {}
    for frame in frames:
        identities = truth[frame][0]
        for identity in identities:
            present.setdefault(int(identity), []).append(frame)

    minimum = max(1, round(min_gap_seconds * fps))
    maximum = max(minimum, round(max_gap_seconds * fps))
    pre_frames = max(1, round(pre_seconds * fps))
    end_padding = round((deadline_seconds + followup_seconds) * fps)
    events = []
    for identity, visible in present.items():
        for before, after in zip(visible, visible[1:]):
            gap_start = before + 1
            gap_frames = after - gap_start
            if gap_frames < minimum or gap_frames > maximum:
                continue
            pre_start = max(visible[0], gap_start - pre_frames)
            if gap_start - pre_start < pre_frames:
                continue
            if after + end_padding > frames[-1]:
                continue
            events.append({
                "person_id": identity,
                "pre_start": pre_start,
                "gap_start": gap_start,
                "return_frame": after,
                "deadline_seconds": deadline_seconds,
                "followup_seconds": followup_seconds,
                "gap_frames": gap_frames,
                "gap_seconds": gap_frames / fps,
                "kind": "unreviewed_annotation_gap",
            })
    return sorted(events, key=lambda event: (-event["gap_frames"], event["gap_start"], event["person_id"]))


def score_episode(event, associations, fps):
    target = event["person_id"]
    pre_frames = range(event["pre_start"], event["gap_start"])
    return_frame = event["return_frame"]
    deadline = return_frame + round(event.get("deadline_seconds", .5) * fps)
    end = deadline + round(event.get("followup_seconds", 2.) * fps)
    if event["gap_start"] <= event["pre_start"] or return_frame <= event["gap_start"]:
        raise ValueError("Episode must contain a pre-gap window and a positive gap")
    if not episode_is_evaluable(event, associations, fps):
        raise ValueError("Scoring window must be completely evaluated, without resetting state")
    history = [associations[frame].get(target) for frame in pre_frames]
    established = [identity for identity in history if identity is not None]
    result = {**event, "gap_seconds": (return_frame - event["gap_start"]) / fps,
              "recovery_latency_seconds": None, "pre_gap_id": None, "delayed_switch": False,
              "wrong_person_id": None, "wrong_person_frame": None}
    if not established:
        return {**result, "outcome": "pre_gap_failure"}
    counts = Counter(established)
    most_common = counts.most_common()
    # A tied pre-gap identity is not a stable starting identity.
    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        return {**result, "outcome": "pre_gap_failure"}
    old = most_common[0][0]
    result["pre_gap_id"] = old
    for frame in range(event["gap_start"], end + 1):
        wrong = [person for person, identity in associations[frame].items() if person != target and identity == old]
        if wrong:
            return {**result, "outcome": "wrong_person_reuse", "wrong_person_id": wrong[0],
                    "wrong_person_frame": frame}
    recovery = [frame for frame in range(return_frame, deadline + 1) if associations[frame].get(target) == old]
    if recovery:
        first = recovery[0]
        changed = any(target in associations[frame] and associations[frame][target] not in (old, None)
                      for frame in range(first, end + 1))
        result["recovery_latency_seconds"] = (first - return_frame) / fps
        result["delayed_switch"] = changed
        if not changed:
            return {**result, "outcome": "correct_recovery"}
    observed_ids = [associations[frame][target] for frame in range(return_frame, deadline + 1) if target in associations[frame]]
    if any(identity is not None for identity in observed_ids):
        return {**result, "outcome": "false_new"}
    if observed_ids:
        return {**result, "outcome": "abstention"}
    return {**result, "outcome": "lost_observation"}


def summarize_episodes(results):
    counts = Counter(result["outcome"] for result in results)
    total = len(results)
    established = total - counts["pre_gap_failure"]
    accepted = counts["correct_recovery"] + counts["wrong_person_reuse"]
    return {"episodes": total, "outcomes": dict(counts), "established_pre_gap": established,
            "unconditional_recovery": counts["correct_recovery"] / total if total else None,
            "conditional_recovery": counts["correct_recovery"] / established if established else None,
            "wrong_link_rate_among_accepted": counts["wrong_person_reuse"] / accepted if accepted else None,
            "accepted_continuity_coverage": accepted / established if established else None,
            "limit": "Event definitions and evaluation associations must be independently reviewed; no intervals from zero/single-sequence samples."}
