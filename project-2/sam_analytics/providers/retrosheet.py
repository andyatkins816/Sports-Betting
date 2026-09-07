"""Strict parser for a pinned Retrosheet 2025 regular-season game log."""

from __future__ import annotations

import csv
import hashlib
import hmac
import io
import os
import re
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

_ARCHIVE_MEMBER = "gl2025.txt"
_EXPECTED_FIELDS = 161
_MAX_ARCHIVE_BYTES = 25 * 1024 * 1024
_MAX_MEMBER_BYTES = 20 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")
_TEAM_CODE = re.compile(r"[A-Z]{3}")


class RetrosheetError(RuntimeError):
    """A Retrosheet archive or row failed strict validation."""


@dataclass(frozen=True)
class RetrosheetProvenance:
    """Exact local evidence needed to trace a normalized game to its source row."""

    provider: str
    archive_sha256: str
    member_name: str
    row_number: int


@dataclass(frozen=True)
class RetrosheetGame:
    """One complete 2025 MLB regular-season result."""

    game_date: date
    game_number: str
    away_team_code: str
    home_team_code: str
    away_score: int
    home_score: int
    completion_info: str | None
    provenance: RetrosheetProvenance


def parse_retrosheet_2025(
    source: bytes | bytearray | memoryview | str | os.PathLike[str],
    *,
    expected_sha256: str,
) -> tuple[RetrosheetGame, ...]:
    """Parse a caller-supplied, SHA-256-pinned Retrosheet ``gl2025.zip`` archive."""
    expected_digest = _validate_expected_digest(expected_sha256)
    archive_bytes = _read_archive(source)
    actual_digest = hashlib.sha256(archive_bytes).hexdigest()
    if not hmac.compare_digest(actual_digest, expected_digest):
        raise RetrosheetError("archive SHA-256 did not match the expected value")

    member_bytes = _read_member(archive_bytes)
    try:
        text = member_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RetrosheetError("gl2025.txt is not valid UTF-8") from exc
    if "\x00" in text:
        raise RetrosheetError("gl2025.txt contains a NUL byte")

    games: list[RetrosheetGame] = []
    identities: set[tuple[date, str, str, str]] = set()
    matchup_numbers: dict[tuple[date, str, str], set[str]] = {}
    try:
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        for row in reader:
            row_number = reader.line_num
            game = _parse_row(row, row_number=row_number, archive_sha256=actual_digest)
            identity = (
                game.game_date,
                game.game_number,
                game.away_team_code,
                game.home_team_code,
            )
            if identity in identities:
                raise RetrosheetError(f"duplicate game identity at row {row_number}")
            identities.add(identity)
            games.append(game)
            matchup = (game.game_date, game.away_team_code, game.home_team_code)
            matchup_numbers.setdefault(matchup, set()).add(game.game_number)
    except csv.Error as exc:
        raise RetrosheetError("gl2025.txt is not valid CSV") from exc

    if not games:
        raise RetrosheetError("gl2025.txt contains no games")
    for numbers in matchup_numbers.values():
        if numbers not in ({"0"}, {"1", "2"}):
            raise RetrosheetError("game-number sequence is ambiguous")
    return tuple(games)


def _validate_expected_digest(expected_sha256: str) -> str:
    if not isinstance(expected_sha256, str) or not _SHA256.fullmatch(expected_sha256):
        raise ValueError("expected_sha256 must be exactly 64 hexadecimal characters")
    return expected_sha256.lower()


def _read_archive(
    source: bytes | bytearray | memoryview | str | os.PathLike[str],
) -> bytes:
    if isinstance(source, (bytes, bytearray, memoryview)):
        archive_bytes = bytes(source)
    elif isinstance(source, (str, os.PathLike)):
        path = Path(source)
        try:
            if path.stat().st_size > _MAX_ARCHIVE_BYTES:
                raise RetrosheetError("Retrosheet archive exceeds the size limit")
            archive_bytes = path.read_bytes()
        except OSError as exc:
            raise RetrosheetError("Retrosheet archive could not be read") from exc
    else:
        raise TypeError("source must be ZIP bytes or a filesystem path")
    if not archive_bytes:
        raise RetrosheetError("Retrosheet archive is empty")
    if len(archive_bytes) > _MAX_ARCHIVE_BYTES:
        raise RetrosheetError("Retrosheet archive exceeds the size limit")
    return archive_bytes


def _read_member(archive_bytes: bytes) -> bytes:
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            members = archive.infolist()
            if len(members) != 1 or members[0].filename != _ARCHIVE_MEMBER:
                raise RetrosheetError("archive must contain only gl2025.txt")
            member = members[0]
            if member.is_dir() or member.flag_bits & 0x1:
                raise RetrosheetError("gl2025.txt must be an unencrypted file")
            if member.file_size > _MAX_MEMBER_BYTES:
                raise RetrosheetError("gl2025.txt exceeds the size limit")
            return archive.read(member)
    except RetrosheetError:
        raise
    except (NotImplementedError, RuntimeError, zipfile.BadZipFile) as exc:
        raise RetrosheetError("Retrosheet archive is not a valid supported ZIP") from exc


def _parse_row(
    row: list[str], *, row_number: int, archive_sha256: str
) -> RetrosheetGame:
    if len(row) != _EXPECTED_FIELDS:
        raise RetrosheetError(f"row {row_number} must contain exactly 161 fields")
    if any("\n" in value or "\r" in value for value in row):
        raise RetrosheetError(f"row {row_number} contains an embedded newline")

    game_date = _parse_date(row[0], row_number=row_number)
    game_number = row[1]
    if game_number not in {"0", "1", "2"}:
        raise RetrosheetError(f"row {row_number} has an invalid 2025 game number")
    if row[4] not in {"AL", "NL"} or row[7] not in {"AL", "NL"}:
        raise RetrosheetError(f"row {row_number} is not an AL/NL regular-season game")

    away_team_code = row[3]
    home_team_code = row[6]
    if not _TEAM_CODE.fullmatch(away_team_code) or not _TEAM_CODE.fullmatch(home_team_code):
        raise RetrosheetError(f"row {row_number} has an invalid Retrosheet team code")
    if away_team_code == home_team_code:
        raise RetrosheetError(f"row {row_number} has identical away and home teams")

    away_score = _parse_score(row[9], row_number=row_number)
    home_score = _parse_score(row[10], row_number=row_number)
    if away_score == home_score:
        raise RetrosheetError(f"row {row_number} has an ambiguous tied result")
    if row[14] or row[15]:
        raise RetrosheetError(f"row {row_number} has a forfeit or protest marker")
    if row[160] != "Y":
        raise RetrosheetError(f"row {row_number} is not marked complete")

    return RetrosheetGame(
        game_date=game_date,
        game_number=game_number,
        away_team_code=away_team_code,
        home_team_code=home_team_code,
        away_score=away_score,
        home_score=home_score,
        completion_info=row[13] or None,
        provenance=RetrosheetProvenance(
            provider="retrosheet",
            archive_sha256=archive_sha256,
            member_name=_ARCHIVE_MEMBER,
            row_number=row_number,
        ),
    )


def _parse_date(value: str, *, row_number: int) -> date:
    if not re.fullmatch(r"2025\d{4}", value):
        raise RetrosheetError(f"row {row_number} has an invalid 2025 date")
    try:
        return date(int(value[:4]), int(value[4:6]), int(value[6:8]))
    except ValueError as exc:
        raise RetrosheetError(f"row {row_number} has an invalid 2025 date") from exc


def _parse_score(value: str, *, row_number: int) -> int:
    if not value.isascii() or not value.isdecimal():
        raise RetrosheetError(f"row {row_number} has an invalid score")
    score = int(value)
    if score > 99:
        raise RetrosheetError(f"row {row_number} has an invalid score")
    return score
