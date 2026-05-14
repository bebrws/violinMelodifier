#!/usr/bin/env python3
"""Create a single-violin melody MIDI from a polyphonic MIDI file.

Uses Magenta note-seq for MIDI parsing, melody inference, and MIDI export.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

try:
    import note_seq
    from note_seq.protobuf import music_pb2
except ImportError as exc:
    raise SystemExit(
        "The note-seq package is required. Install it with: python3 -m pip install note-seq"
    ) from exc


VIOLIN_PROGRAM = 40


def copy_repeated(source, target) -> None:
    for item in source:
        target.add().CopyFrom(item)


def base_output_sequence(source: music_pb2.NoteSequence) -> music_pb2.NoteSequence:
    output = music_pb2.NoteSequence()
    output.id = f"{source.id or source.filename}:melody-violin"
    output.filename = source.filename
    output.total_time = source.total_time
    output.ticks_per_quarter = source.ticks_per_quarter

    copy_repeated(source.tempos, output.tempos)
    copy_repeated(source.time_signatures, output.time_signatures)
    copy_repeated(source.key_signatures, output.key_signatures)

    return output


def add_note(
    output: music_pb2.NoteSequence,
    source_note: music_pb2.NoteSequence.Note,
    *,
    velocity: int | None,
) -> None:
    note = output.notes.add()
    note.pitch = source_note.pitch
    note.start_time = source_note.start_time
    note.end_time = source_note.end_time
    note.velocity = velocity if velocity is not None else source_note.velocity
    note.instrument = 0
    note.program = VIOLIN_PROGRAM
    note.is_drum = False


def infer_melody_sequence(
    source: music_pb2.NoteSequence,
    *,
    velocity: int | None,
) -> music_pb2.NoteSequence:
    working = music_pb2.NoteSequence()
    working.CopyFrom(source)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        melody_instrument = note_seq.infer_melody_for_sequence(working)

    output = base_output_sequence(source)
    melody_notes = [
        note
        for note in working.notes
        if note.instrument == melody_instrument and not note.is_drum
    ]
    melody_notes.sort(key=lambda note: (note.start_time, note.end_time, note.pitch))

    for note in melody_notes:
        add_note(output, note, velocity=velocity)

    output.total_time = max(
        [source.total_time] + [note.end_time for note in output.notes],
        default=source.total_time,
    )
    return output


def highest_onset_melody_sequence(
    source: music_pb2.NoteSequence,
    *,
    velocity: int | None,
) -> music_pb2.NoteSequence:
    output = base_output_sequence(source)
    note_groups: dict[float, list[music_pb2.NoteSequence.Note]] = {}

    for note in source.notes:
        if note.is_drum:
            continue
        note_groups.setdefault(round(note.start_time, 6), []).append(note)

    selected_notes = []
    for start_time in sorted(note_groups):
        selected = max(
            note_groups[start_time],
            key=lambda note: (note.pitch, note.end_time - note.start_time, note.velocity),
        )
        selected_notes.append(selected)

    for index, note in enumerate(selected_notes):
        copied = music_pb2.NoteSequence.Note()
        copied.CopyFrom(note)
        if index + 1 < len(selected_notes):
            copied.end_time = min(copied.end_time, selected_notes[index + 1].start_time)
        if copied.end_time > copied.start_time:
            add_note(output, copied, velocity=velocity)

    output.total_time = max(
        [source.total_time] + [note.end_time for note in output.notes],
        default=source.total_time,
    )
    return output


def write_melody_violin(
    midi_file: Path,
    output_file: Path,
    *,
    method: str,
    steps_per_quarter: int,
    qpm: float | None,
    velocity: int | None,
) -> music_pb2.NoteSequence:
    source = note_seq.midi_file_to_note_sequence(str(midi_file))

    if method == "infer":
        output = infer_melody_sequence(source, velocity=velocity)
    elif method == "highest":
        output = highest_onset_melody_sequence(source, velocity=velocity)
    else:
        raise ValueError(f"Unknown method: {method}")

    if not output.notes:
        raise ValueError(f"No melody notes found in {midi_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    note_seq.note_sequence_to_midi_file(output, str(output_file))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a MIDI file containing one violin carrying the melody."
    )
    parser.add_argument("input_midi", type=Path)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output MIDI path. Defaults to '<input-stem>_melody_violin.mid'.",
    )
    parser.add_argument(
        "--method",
        choices=("infer", "highest"),
        default="infer",
        help=(
            "'infer' uses note-seq's melody inference. "
            "'highest' keeps the highest pitched note at each note onset."
        ),
    )
    parser.add_argument(
        "--steps-per-quarter",
        type=int,
        default=4,
        help="Reserved for future quantized extraction modes.",
    )
    parser.add_argument(
        "--qpm",
        type=float,
        help="Reserved for future quantized extraction modes.",
    )
    parser.add_argument(
        "--velocity",
        type=int,
        choices=range(1, 128),
        metavar="1-127",
        help="Force all output note velocities to this value.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output or args.input_midi.with_name(
        f"{args.input_midi.stem}_melody_violin.mid"
    )

    try:
        sequence = write_melody_violin(
            args.input_midi,
            output,
            method=args.method,
            steps_per_quarter=args.steps_per_quarter,
            qpm=args.qpm,
            velocity=args.velocity,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(output)
    print(f"notes={len(sequence.notes)}")
    print(f"duration={sequence.total_time:.3f}s")
    print("program=40")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
