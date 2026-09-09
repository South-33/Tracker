"""Score explicitly defined disappearance episodes from evaluation associations.

Each frame maps ground-truth IDs to a public ID, None for observed-but-abstained,
or has no entry for a missed observation. Association is evaluation-only.
"""
from collections import Counter


def score_episode(event, associations, fps):
    target = event["person_id"]
    pre_frames = range(event["pre_start"], event["gap_start"])
    return_frame = event["return_frame"]
    deadline = return_frame + round(event.get("deadline_seconds", .5) * fps)
    end = deadline + round(event.get("followup_seconds", 2.) * fps)
    if event["gap_start"] <= event["pre_start"] or return_frame <= event["gap_start"]:
        raise ValueError("Episode must contain a pre-gap window and a positive gap")
    if any(frame not in associations for frame in range(event["pre_start"], end + 1)):
        raise ValueError("Scoring window must be completely evaluated, without resetting state")
    history = [associations[frame].get(target) for frame in pre_frames]
    established = [identity for identity in history if identity is not None]
    result = {**event, "gap_seconds": (return_frame - event["gap_start"]) / fps,
              "recovery_latency_seconds": None, "pre_gap_id": None, "delayed_switch": False}
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
        if any(person != target and identity == old for person, identity in associations[frame].items()):
            return {**result, "outcome": "wrong_person_reuse"}
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
