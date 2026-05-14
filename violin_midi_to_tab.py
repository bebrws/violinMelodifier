#!/usr/bin/env python3
"""Convert a violin MIDI file to first-position ASCII tab."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
import math
from pathlib import Path
import sys

try:
    import mido
except ImportError as exc:
    raise SystemExit(
        "The mido package is required. Install it with: python3 -m pip install mido"
    ) from exc


NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
UNPLACED_SCORE = 250.0


@dataclass(frozen=True)
class ViolinString:
    name: str
    open_pitch: int


@dataclass(frozen=True)
class FingerPlacement:
    offset: int
    finger: int
    label: str
    frame: str
    base_score: float
    warning: str | None = None


@dataclass(frozen=True)
class MidiNote:
    pitch: int
    start_tick: int
    end_tick: int
    velocity: int
    channel: int
    track: int


@dataclass(frozen=True)
class MidiMetadata:
    ticks_per_beat: int
    track_count: int
    time_signature: tuple[int, int]
    tempo: int
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class Candidate:
    note: MidiNote
    string: ViolinString
    string_index: int
    placement: FingerPlacement


@dataclass(frozen=True)
class TabAssignment:
    note: MidiNote
    candidate: Candidate | None
    start_step: int
    end_step: int


STRINGS = (
    ViolinString("G", 55),
    ViolinString("D", 62),
    ViolinString("A", 69),
    ViolinString("E", 76),
)

FINGER_PLACEMENTS = {
    0: FingerPlacement(0, 0, "0", "open string", 0.2),
    1: FingerPlacement(
        1,
        1,
        "L1",
        "low-first-finger frame",
        4.0,
        "low first finger; usable in first position but less basic than regular frames",
    ),
    2: FingerPlacement(2, 1, "1", "regular first-finger frame", 0.7),
    3: FingerPlacement(3, 2, "L2", "low-second-finger frame", 0.8),
    4: FingerPlacement(4, 2, "H2", "high-second-finger frame", 0.7),
    5: FingerPlacement(5, 3, "3", "third-finger anchor frame", 0.7),
    6: FingerPlacement(
        6,
        3,
        "H3",
        "chromatic high-third frame",
        5.0,
        "high third finger; this is a chromatic reach and may be awkward for beginners",
    ),
    7: FingerPlacement(7, 4, "4", "fourth-finger first-position frame", 2.8),
}

STRICT_OFFSETS = {0, 2, 3, 4, 5, 7}
BASIC_FRAME_SETS = (
    frozenset((0, 2, 4, 5, 7)),
    frozenset((0, 2, 3, 5, 7)),
    frozenset((0, 1, 3, 5, 7)),
)


def note_name(pitch: int) -> str:
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def read_midi_notes(midi_file: Path) -> tuple[list[MidiNote], MidiMetadata]:
    midi = mido.MidiFile(midi_file)
    notes: list[MidiNote] = []
    warnings: list[str] = []
    tempos: list[tuple[int, int]] = []
    time_signatures: list[tuple[int, int, int]] = []

    for track_index, track in enumerate(midi.tracks):
        tick = 0
        active: dict[tuple[int, int], list[tuple[int, int]]] = {}

        for message in track:
            tick += message.time

            if message.type == "set_tempo":
                tempos.append((tick, message.tempo))
            elif message.type == "time_signature":
                time_signatures.append((tick, message.numerator, message.denominator))
            elif message.type == "note_on" and message.velocity > 0:
                key = (message.channel, message.note)
                active.setdefault(key, []).append((tick, message.velocity))
            elif message.type in ("note_off", "note_on"):
                channel = getattr(message, "channel", None)
                pitch = getattr(message, "note", None)
                if channel is None or pitch is None:
                    continue
                key = (channel, pitch)
                starts = active.get(key)
                if not starts:
                    warnings.append(
                        f"track {track_index}: unmatched note off for {note_name(pitch)} "
                        f"at tick {tick}"
                    )
                    continue
                start_tick, velocity = starts.pop(0)
                if tick > start_tick:
                    notes.append(
                        MidiNote(
                            pitch=pitch,
                            start_tick=start_tick,
                            end_tick=tick,
                            velocity=velocity,
                            channel=channel,
                            track=track_index,
                        )
                    )

        for (channel, pitch), starts in active.items():
            for start_tick, _velocity in starts:
                warnings.append(
                    f"track {track_index}: unterminated {note_name(pitch)} on channel "
                    f"{channel + 1} starting at tick {start_tick}"
                )

    notes.sort(key=lambda note: (note.start_tick, note.pitch, note.end_tick))
    time_signatures.sort(key=lambda item: item[0])
    tempos.sort(key=lambda item: item[0])

    metadata = MidiMetadata(
        ticks_per_beat=midi.ticks_per_beat,
        track_count=len(midi.tracks),
        time_signature=(
            (time_signatures[0][1], time_signatures[0][2])
            if time_signatures
            else (4, 4)
        ),
        tempo=tempos[0][1] if tempos else mido.bpm2tempo(120),
        warnings=tuple(warnings),
    )
    return notes, metadata


def quantize_tick(tick: int, ticks_per_beat: int, grid: int) -> int:
    value = Fraction(tick * grid, ticks_per_beat)
    return int(value + Fraction(1, 2))


def beat_text(tick: int, ticks_per_beat: int) -> str:
    beat = Fraction(tick, ticks_per_beat) + 1
    text = f"{float(beat):.3f}"
    return text.rstrip("0").rstrip(".")


def duration_text(note: MidiNote, ticks_per_beat: int) -> str:
    duration = Fraction(note.end_tick - note.start_tick, ticks_per_beat)
    text = f"{float(duration):.3f}"
    return text.rstrip("0").rstrip(".")


def candidates_for_note(note: MidiNote, *, strict_basic_frames: bool) -> list[Candidate]:
    candidates: list[Candidate] = []
    for string_index, string in enumerate(STRINGS):
        offset = note.pitch - string.open_pitch
        placement = FINGER_PLACEMENTS.get(offset)
        if placement is None:
            continue
        if strict_basic_frames and offset not in STRICT_OFFSETS:
            continue
        candidates.append(
            Candidate(
                note=note,
                string=string,
                string_index=string_index,
                placement=placement,
            )
        )
    return candidates


def assignment_base_score(assignment: TabAssignment) -> float:
    if assignment.candidate is None:
        return UNPLACED_SCORE
    return assignment.candidate.placement.base_score


def same_string_frame_cost(previous: Candidate, current: Candidate) -> float:
    if previous.string_index != current.string_index:
        return 0.0

    previous_offset = previous.placement.offset
    current_offset = current.placement.offset
    if previous_offset == current_offset:
        return -0.3

    pair = frozenset((previous_offset, current_offset))
    if any(pair.issubset(frame) for frame in BASIC_FRAME_SETS):
        return 0.0

    cost = 1.5
    if 6 in pair:
        cost += 2.5
    if abs(previous_offset - current_offset) > 5:
        cost += 3.0
    return cost


def transition_score(
    previous_group: tuple[TabAssignment, ...],
    current_group: tuple[TabAssignment, ...],
) -> float:
    previous = [assignment for assignment in previous_group if assignment.candidate]
    current = [assignment for assignment in current_group if assignment.candidate]
    if not previous or not current:
        return 0.0

    previous_melody = max(previous, key=lambda assignment: assignment.note.pitch)
    current_melody = max(current, key=lambda assignment: assignment.note.pitch)
    previous_candidate = previous_melody.candidate
    current_candidate = current_melody.candidate
    assert previous_candidate is not None
    assert current_candidate is not None

    pitch_delta = abs(previous_melody.note.pitch - current_melody.note.pitch)
    string_delta = abs(previous_candidate.string_index - current_candidate.string_index)
    score = string_delta * 0.8

    if pitch_delta <= 2 and string_delta:
        score += 1.2
    if previous_candidate.string_index == current_candidate.string_index:
        score += same_string_frame_cost(previous_candidate, current_candidate)
    return score


def group_internal_score(group: tuple[TabAssignment, ...]) -> float:
    score = sum(assignment_base_score(assignment) for assignment in group)
    placed = [assignment for assignment in group if assignment.candidate]
    if len(placed) > 1:
        strings = [assignment.candidate.string_index for assignment in placed]
        assert all(assignment.candidate is not None for assignment in placed)
        span = max(strings) - min(strings)
        if span > len(placed) - 1:
            score += (span - len(placed) + 1) * 4.0
        if len(placed) > 2:
            score += (len(placed) - 2) * 3.0
    return score


def build_group_options(
    notes: list[MidiNote],
    *,
    ticks_per_beat: int,
    grid: int,
    strict_basic_frames: bool,
) -> list[tuple[TabAssignment, ...]]:
    option_lists: list[list[TabAssignment]] = []
    for note in notes:
        start_step = quantize_tick(note.start_tick, ticks_per_beat, grid)
        end_step = max(
            start_step + 1,
            quantize_tick(note.end_tick, ticks_per_beat, grid),
        )
        options = [
            TabAssignment(note, candidate, start_step, end_step)
            for candidate in candidates_for_note(
                note, strict_basic_frames=strict_basic_frames
            )
        ]
        options.append(TabAssignment(note, None, start_step, end_step))
        options.sort(key=assignment_base_score)
        option_lists.append(options[:5])

    combinations: list[tuple[TabAssignment, ...]] = []
    for combination in product(*option_lists):
        string_indexes = [
            assignment.candidate.string_index
            for assignment in combination
            if assignment.candidate is not None
        ]
        if len(string_indexes) != len(set(string_indexes)):
            continue
        combinations.append(tuple(combination))

    combinations.sort(key=group_internal_score)
    return combinations[:100]


def choose_assignments(
    notes: list[MidiNote],
    *,
    ticks_per_beat: int,
    grid: int,
    strict_basic_frames: bool,
) -> list[TabAssignment]:
    grouped: dict[int, list[MidiNote]] = {}
    for note in notes:
        start_step = quantize_tick(note.start_tick, ticks_per_beat, grid)
        grouped.setdefault(start_step, []).append(note)

    scores: dict[tuple[TabAssignment, ...], float] = {(): 0.0}
    paths: dict[
        tuple[TabAssignment, ...], list[tuple[TabAssignment, ...]]
    ] = {(): []}

    for start_step in sorted(grouped):
        group_notes = sorted(grouped[start_step], key=lambda note: note.pitch)
        options = build_group_options(
            group_notes,
            ticks_per_beat=ticks_per_beat,
            grid=grid,
            strict_basic_frames=strict_basic_frames,
        )
        next_scores: dict[tuple[TabAssignment, ...], float] = {}
        next_paths: dict[
            tuple[TabAssignment, ...], list[tuple[TabAssignment, ...]]
        ] = {}

        for previous_group, previous_score in scores.items():
            for option in options:
                score = (
                    previous_score
                    + group_internal_score(option)
                    + transition_score(previous_group, option)
                )
                if score < next_scores.get(option, math.inf):
                    next_scores[option] = score
                    next_paths[option] = paths[previous_group] + [option]

        kept = sorted(next_scores, key=next_scores.get)[:80]
        scores = {group: next_scores[group] for group in kept}
        paths = {group: next_paths[group] for group in kept}

    if not scores:
        return []

    best_group = min(scores, key=scores.get)
    return [
        assignment
        for group in paths[best_group]
        for assignment in group
    ]


def steps_per_measure(time_signature: tuple[int, int], grid: int) -> int:
    numerator, denominator = time_signature
    steps = Fraction(numerator * 4 * grid, denominator)
    return max(1, int(steps + Fraction(1, 2)))


def token_cell(token: str, width: int) -> str:
    if len(token) >= width:
        return token[:width]
    left = (width - len(token)) // 2
    right = width - len(token) - left
    return ("-" * left) + token + ("-" * right)


def render_count_line(
    segment_start: int,
    segment_end: int,
    *,
    grid: int,
    measure_steps: int,
    cell_width: int,
) -> str:
    line = "  |"
    for step in range(segment_start, segment_end):
        if step != segment_start and step % measure_steps == 0:
            line += "|"
        if step % grid == 0:
            beat = (step % measure_steps) // grid + 1
            line += str(beat).center(cell_width)
        else:
            line += " " * cell_width
    return line + "|"


def render_tab(
    assignments: list[TabAssignment],
    *,
    time_signature: tuple[int, int],
    grid: int,
    measures_per_line: int,
    cell_width: int,
) -> list[str]:
    placed = [assignment for assignment in assignments if assignment.candidate]
    max_step = max((assignment.end_step for assignment in assignments), default=0)
    if not placed:
        return ["(no notes could be placed in first position)"]

    measure_steps = steps_per_measure(time_signature, grid)
    segment_steps = measure_steps * measures_per_line
    padded_total_steps = int(math.ceil(max_step / measure_steps) * measure_steps)

    onsets: dict[tuple[int, int], TabAssignment] = {}
    sustains: set[tuple[int, int]] = set()
    for assignment in placed:
        assert assignment.candidate is not None
        key = (assignment.start_step, assignment.candidate.string_index)
        onsets[key] = assignment
        for step in range(assignment.start_step + 1, assignment.end_step):
            sustains.add((step, assignment.candidate.string_index))

    lines: list[str] = []
    for segment_start in range(0, padded_total_steps, segment_steps):
        segment_end = min(segment_start + segment_steps, padded_total_steps)
        first_measure = segment_start // measure_steps + 1
        last_measure = (segment_end - 1) // measure_steps + 1
        lines.append(f"Measures {first_measure}-{last_measure}")
        lines.append(
            render_count_line(
                segment_start,
                segment_end,
                grid=grid,
                measure_steps=measure_steps,
                cell_width=cell_width,
            )
        )

        for string_index in reversed(range(len(STRINGS))):
            string = STRINGS[string_index]
            line = f"{string.name} |"
            for step in range(segment_start, segment_end):
                if step != segment_start and step % measure_steps == 0:
                    line += "|"
                assignment = onsets.get((step, string_index))
                if assignment and assignment.candidate:
                    line += token_cell(assignment.candidate.placement.label, cell_width)
                elif (step, string_index) in sustains:
                    line += "=" * cell_width
                else:
                    line += "-" * cell_width
            lines.append(line + "|")
        lines.append("")
    return lines


def collect_warnings(
    assignments: list[TabAssignment],
    metadata: MidiMetadata,
    *,
    ticks_per_beat: int,
) -> list[str]:
    warnings = list(metadata.warnings)
    for assignment in assignments:
        pitch = note_name(assignment.note.pitch)
        beat = beat_text(assignment.note.start_tick, ticks_per_beat)
        if assignment.candidate is None:
            warnings.append(
                f"beat {beat}: {pitch} cannot be placed in the selected first-position "
                "finger frames"
            )
            continue
        placement_warning = assignment.candidate.placement.warning
        if placement_warning:
            warnings.append(f"beat {beat}: {pitch} uses {placement_warning}")
    return warnings


def fingering_detail_lines(
    assignments: list[TabAssignment],
    *,
    ticks_per_beat: int,
) -> list[str]:
    lines = [
        "Fingering detail",
        "beat     dur     note  tab    frame",
        "-------  ------  ----  -----  -------------------------------",
    ]
    for assignment in sorted(
        assignments,
        key=lambda item: (item.note.start_tick, item.note.pitch, item.note.end_tick),
    ):
        beat = beat_text(assignment.note.start_tick, ticks_per_beat)
        duration = duration_text(assignment.note, ticks_per_beat)
        pitch = note_name(assignment.note.pitch)
        if assignment.candidate is None:
            tab = "??"
            frame = "outside selected first-position frames"
        else:
            tab = (
                f"{assignment.candidate.string.name}:"
                f"{assignment.candidate.placement.label}"
            )
            frame = assignment.candidate.placement.frame
        lines.append(f"{beat:<7}  {duration:<6}  {pitch:<4}  {tab:<5}  {frame}")
    return lines


def build_report(
    *,
    input_file: Path,
    notes: list[MidiNote],
    assignments: list[TabAssignment],
    metadata: MidiMetadata,
    grid: int,
    measures_per_line: int,
    cell_width: int,
    strict_basic_frames: bool,
) -> str:
    numerator, denominator = metadata.time_signature
    bpm = mido.tempo2bpm(metadata.tempo)
    placed_count = sum(1 for assignment in assignments if assignment.candidate)
    warnings = collect_warnings(
        assignments,
        metadata,
        ticks_per_beat=metadata.ticks_per_beat,
    )

    lines = [
        "Violin ASCII Tab",
        f"Input: {input_file}",
        f"Tracks: {metadata.track_count}",
        f"Tempo: {bpm:.2f} bpm",
        f"Time signature: {numerator}/{denominator}",
        f"Quantization: {grid} tab steps per quarter note",
        f"Notes: {len(notes)} total, {placed_count} placed, "
        f"{len(notes) - placed_count} unplaced",
        "",
        "Legend",
        "0=open string, 1=regular first, L1=low first, L2=low second,",
        "H2=high second, 3=third, H3=high third, 4=fourth finger.",
        "String lines are shown high to low: E, A, D, G.",
        "Sustain cells are shown as ====.",
        "",
        "Fingering goal",
        "The scorer stays in first position, prefers open strings and common",
        "low-second/high-second finger frames, and penalizes chromatic reaches,",
        "large same-string finger spreads, and unnecessary string crossings.",
    ]
    if strict_basic_frames:
        lines.append("Strict mode is on: L1 and H3 placements are rejected.")

    lines.extend(["", "Tab", ""])
    lines.extend(
        render_tab(
            assignments,
            time_signature=metadata.time_signature,
            grid=grid,
            measures_per_line=measures_per_line,
            cell_width=cell_width,
        )
    )
    lines.extend(
        [
            "",
            *fingering_detail_lines(
                assignments,
                ticks_per_beat=metadata.ticks_per_beat,
            ),
        ]
    )

    if warnings:
        lines.extend(["", "Warnings"])
        max_warning_lines = 40
        for warning in warnings[:max_warning_lines]:
            lines.append(f"- {warning}")
        if len(warnings) > max_warning_lines:
            lines.append(f"- ... {len(warnings) - max_warning_lines} more warnings")

    return "\n".join(lines) + "\n"


def convert_midi_to_tab(
    input_midi: Path,
    output_file: Path,
    *,
    grid: int,
    measures_per_line: int,
    cell_width: int,
    strict_basic_frames: bool,
) -> tuple[int, int]:
    notes, metadata = read_midi_notes(input_midi)
    if not notes:
        raise ValueError(f"No MIDI notes found in {input_midi}")

    assignments = choose_assignments(
        notes,
        ticks_per_beat=metadata.ticks_per_beat,
        grid=grid,
        strict_basic_frames=strict_basic_frames,
    )
    report = build_report(
        input_file=input_midi,
        notes=notes,
        assignments=assignments,
        metadata=metadata,
        grid=grid,
        measures_per_line=measures_per_line,
        cell_width=cell_width,
        strict_basic_frames=strict_basic_frames,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(report, encoding="utf-8")
    placed_count = sum(1 for assignment in assignments if assignment.candidate)
    return len(notes), placed_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a violin MIDI file to first-position ASCII tab in a .txt file."
        )
    )
    parser.add_argument("input_midi", type=Path)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output txt path. Defaults to '<input-stem>_violin_tab.txt'.",
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=4,
        help="Tab steps per quarter note. 4 gives a sixteenth-note grid.",
    )
    parser.add_argument(
        "--measures-per-line",
        type=int,
        default=4,
        help="How many measures to render per tab system.",
    )
    parser.add_argument(
        "--cell-width",
        type=int,
        default=4,
        help="Fixed ASCII width for each quantized tab cell.",
    )
    parser.add_argument(
        "--strict-basic-frames",
        action="store_true",
        help="Reject low-first and high-third chromatic placements instead of warning.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.grid < 1:
        print("error: --grid must be at least 1", file=sys.stderr)
        return 1
    if args.measures_per_line < 1:
        print("error: --measures-per-line must be at least 1", file=sys.stderr)
        return 1
    if args.cell_width < 3:
        print("error: --cell-width must be at least 3", file=sys.stderr)
        return 1

    output = args.output or args.input_midi.with_name(
        f"{args.input_midi.stem}_violin_tab.txt"
    )

    try:
        total_notes, placed_notes = convert_midi_to_tab(
            args.input_midi,
            output,
            grid=args.grid,
            measures_per_line=args.measures_per_line,
            cell_width=args.cell_width,
            strict_basic_frames=args.strict_basic_frames,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(output)
    print(f"notes={total_notes}")
    print(f"placed={placed_notes}")
    print(f"unplaced={total_notes - placed_notes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
