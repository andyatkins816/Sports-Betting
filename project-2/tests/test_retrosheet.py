import csv
import hashlib
import io
import tempfile
import unittest
import zipfile
from dataclasses import FrozenInstanceError
from datetime import date
from pathlib import Path

from sam_analytics.providers.retrosheet import RetrosheetError, parse_retrosheet_2025


class RetrosheetProviderTests(unittest.TestCase):
    def test_parses_pinned_bytes_and_preserves_codes_and_provenance(self):
        archive = _archive([_row()])
        digest = hashlib.sha256(archive).hexdigest()

        games = parse_retrosheet_2025(archive, expected_sha256=digest.upper())

        self.assertIsInstance(games, tuple)
        self.assertEqual(len(games), 1)
        game = games[0]
        self.assertEqual(game.game_date, date(2025, 3, 27))
        self.assertEqual(game.game_number, "0")
        self.assertEqual(game.away_team_code, "CHN")
        self.assertEqual(game.home_team_code, "ARI")
        self.assertEqual(game.away_score, 10)
        self.assertEqual(game.home_score, 6)
        self.assertIsNone(game.completion_info)
        self.assertEqual(game.provenance.provider, "retrosheet")
        self.assertEqual(game.provenance.archive_sha256, digest)
        self.assertEqual(game.provenance.member_name, "gl2025.txt")
        self.assertEqual(game.provenance.row_number, 1)
        with self.assertRaises(FrozenInstanceError):
            game.home_score = 7

    def test_accepts_a_caller_supplied_zip_path(self):
        archive = _archive([_row()])
        digest = hashlib.sha256(archive).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pinned-retrosheet.zip"
            path.write_bytes(archive)

            games = parse_retrosheet_2025(path, expected_sha256=digest)

        self.assertEqual(len(games), 1)

    def test_marks_suspended_game_completion_information_for_safe_exclusion(self):
        completion_info = "20250418,CHI11,3,2,1"
        archive = _archive([_row(**{"13": completion_info})])

        game = parse_retrosheet_2025(
            archive,
            expected_sha256=hashlib.sha256(archive).hexdigest(),
        )[0]

        self.assertEqual(game.completion_info, completion_info)

    def test_rejects_unpinned_or_malformed_archives(self):
        archive = _archive([_row()])
        cases = [
            (
                "wrong digest",
                archive,
                "0" * 64,
                "SHA-256 did not match",
            ),
            (
                "not a zip",
                b"not-a-zip",
                hashlib.sha256(b"not-a-zip").hexdigest(),
                "not a valid supported ZIP",
            ),
            (
                "wrong member",
                _archive([_row()], member="other.txt"),
                None,
                "only gl2025.txt",
            ),
            (
                "extra member",
                _archive([_row()], extra_member=True),
                None,
                "only gl2025.txt",
            ),
        ]
        for name, payload, expected_digest, error in cases:
            with self.subTest(name=name):
                digest = expected_digest or hashlib.sha256(payload).hexdigest()
                with self.assertRaisesRegex(RetrosheetError, error):
                    parse_retrosheet_2025(payload, expected_sha256=digest)
        with self.assertRaisesRegex(ValueError, "64 hexadecimal"):
            parse_retrosheet_2025(archive, expected_sha256="not-a-digest")

    def test_rejects_incomplete_or_malformed_rows(self):
        cases = {
            "wrong field count": (_row()[:-1], "exactly 161 fields"),
            "wrong year": (_row(**{"0": "20240327"}), "invalid 2025 date"),
            "impossible date": (_row(**{"0": "20250229"}), "invalid 2025 date"),
            "unknown game number": (_row(**{"1": "3"}), "invalid 2025 game number"),
            "non-MLB league": (_row(**{"4": "AA"}), "not an AL/NL"),
            "malformed team code": (_row(**{"3": "Chi"}), "invalid Retrosheet team code"),
            "same teams": (_row(**{"6": "CHN"}), "identical away and home"),
            "negative score": (_row(**{"9": "-1"}), "invalid score"),
            "tied score": (_row(**{"9": "6"}), "ambiguous tied result"),
            "forfeit": (_row(**{"14": "V"}), "forfeit or protest"),
            "partial record": (_row(**{"160": "P"}), "not marked complete"),
        }
        for name, (row, error) in cases.items():
            with self.subTest(name=name):
                archive = _archive([row])
                with self.assertRaisesRegex(RetrosheetError, error):
                    parse_retrosheet_2025(
                        archive, expected_sha256=hashlib.sha256(archive).hexdigest()
                    )

    def test_rejects_duplicate_and_ambiguous_game_numbers(self):
        cases = [
            ([_row(), _row()], "duplicate game identity"),
            ([_row(**{"1": "1"})], "game-number sequence is ambiguous"),
            (
                [
                    _row(**{"1": "1"}),
                    _row(**{"1": "2", "9": "7", "10": "4"}),
                ],
                None,
            ),
        ]
        for rows, error in cases:
            with self.subTest(error=error):
                archive = _archive(rows)
                digest = hashlib.sha256(archive).hexdigest()
                if error is None:
                    self.assertEqual(len(parse_retrosheet_2025(archive, expected_sha256=digest)), 2)
                else:
                    with self.assertRaisesRegex(RetrosheetError, error):
                        parse_retrosheet_2025(archive, expected_sha256=digest)


def _row(**overrides: str) -> list[str]:
    row = [""] * 161
    row[0] = "20250327"
    row[1] = "0"
    row[3] = "CHN"
    row[4] = "NL"
    row[6] = "ARI"
    row[7] = "NL"
    row[9] = "10"
    row[10] = "6"
    row[160] = "Y"
    for index, value in overrides.items():
        row[int(index)] = value
    return row


def _archive(
    rows: list[list[str]], *, member: str = "gl2025.txt", extra_member: bool = False
) -> bytes:
    csv_bytes = io.StringIO(newline="")
    csv.writer(csv_bytes, lineterminator="\n").writerows(rows)
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, csv_bytes.getvalue().encode())
        if extra_member:
            archive.writestr("unexpected.txt", b"unexpected")
    return archive_bytes.getvalue()


if __name__ == "__main__":
    unittest.main()
