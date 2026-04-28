"""Post all MOS trials to a Telegram channel with native anonymous polls.

For each of 11 species, posts:
    1. Species header + reference audio (~35 s)
    2. Four samples, each followed by a 1-5 MOS poll

After all trials are posted, a mapping from poll_id -> trial_id is saved
to  subjective_eval/poll_mapping.json  so results can be collected later.

Usage:
    1. Create a bot via @BotFather, get the token.
    2. Add the bot as admin to your channel.
    3. Run:

        export TELEGRAM_BOT_TOKEN="..."
        export TELEGRAM_CHANNEL_ID="@your_channel"
        uv run python subjective_eval/telegram_survey.py

    Use --delay to control pause between posts (default 2 s).
    Use --dry-run to verify files without posting.
    Use --collect to stop polls and download results.
"""

import argparse
import asyncio
import csv
import json
import logging
import os
from pathlib import Path

from telegram import Bot

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent

SPECIES_COMMON_NAMES = {
    "baffal1": "Barred Forest-Falcon",
    "bobfly1": "Boat-billed Flycatcher",
    "coffal1": "Collared Forest-Falcon",
    "compot1": "Common Potoo",
    "greant1": "Great Antshrike",
    "pirfly1": "Piratic Flycatcher",
    "roahaw": "Roadside Hawk",
    "socfly1": "Social Flycatcher",
    "trokin": "Tropical Kingbird",
    "trsowl": "Tropical Screech-Owl",
    "whtdov": "White-tipped Dove",
}

POLL_OPTIONS = [
    "1 - Дуже погано",
    "2 - Погано",
    "3 - Задовільно",
    "4 - Добре",
    "5 - Відмінно",
]

POLL_VALUE_MAP = {i: v for i, v in enumerate(range(1, 6))}


# ---------------------------------------------------------------------------
# Trial loading
# ---------------------------------------------------------------------------


def load_trials(csv_path):
    trials = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            row["trial_id"] = int(row["trial_id"])
            trials.append(row)
    return trials


def group_by_species(trials):
    """Group trials into species blocks preserving order."""
    blocks = []
    current_code = None
    current_block = None
    for t in trials:
        if t["ebird_code"] != current_code:
            current_code = t["ebird_code"]
            current_block = {
                "ebird_code": current_code,
                "common_name": SPECIES_COMMON_NAMES.get(current_code, current_code),
                "ref_path": t["ref_path"],
                "trials": [],
            }
            blocks.append(current_block)
        current_block["trials"].append(t)
    return blocks


# ---------------------------------------------------------------------------
# Post to channel
# ---------------------------------------------------------------------------


async def post_instructions(bot: Bot, channel_id: str):
    text = (
        "Bird Vocalization — MOS Listening Test\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "For each bird species you will find:\n"
        "  1. A reference recording (~35 s) for context\n"
        "  2. Four audio samples to rate\n\n"
        "Rate each sample on how natural it sounds:\n"
        "  1 = Bad\n"
        "  2 = Poor\n"
        "  3 = Fair\n"
        "  4 = Good\n"
        "  5 = Excellent\n\n"
        "Use headphones if possible. Replay clips as many "
        "times as you like before voting.\n\n"
        "Thank you for participating!"
    )
    await bot.send_message(chat_id=channel_id, text=text)


async def post_species_block(
    bot: Bot,
    channel_id: str,
    block: dict,
    block_idx: int,
    total_blocks: int,
    global_offset: int,
    total_trials: int,
    delay: float,
):
    code = block["ebird_code"]
    common = block["common_name"]

    separator = "— " * 20
    header = (
        f"{separator}\n"
        f"Вид пташки (англійською) {block_idx}/{total_blocks}: {common}\n"
        f"{separator}"
    )
    await bot.send_message(chat_id=channel_id, text=header)

    ref_path = SCRIPT_DIR / block["ref_path"]
    with open(ref_path, "rb") as f:
        await bot.send_audio(
            chat_id=channel_id,
            audio=f,
            title=f"{common}",
            caption="Приклад оригінального запису",
        )

    if delay > 0:
        await asyncio.sleep(delay)

    entries = []
    for i, trial in enumerate(block["trials"]):
        tid = trial["trial_id"]
        trial_num = global_offset + i + 1

        sample_path = SCRIPT_DIR / trial["sample_path"]
        with open(sample_path, "rb") as f:
            await bot.send_audio(
                chat_id=channel_id,
                audio=f,
                title=f"Запис {trial_num}",
                caption=f"Запис {trial_num}/{total_trials}",
            )

        poll_msg = await bot.send_poll(
            chat_id=channel_id,
            question=(f"Кандидат в опитуванні {trial_num}/{total_trials}: звучить ..."),
            options=POLL_OPTIONS,
            is_anonymous=True,
        )

        entries.append(
            {
                "trial_id": tid,
                "poll_id": poll_msg.poll.id,
                "message_id": poll_msg.message_id,
                "ebird_code": code,
                "system": trial["system"],
            }
        )

        if delay > 0:
            await asyncio.sleep(delay)

    return entries


async def post_all(
    token: str,
    channel_id: str,
    csv_path: Path,
    delay: float,
    resume_from: int,
    dry_run: bool,
):
    trials = load_trials(csv_path)
    blocks = group_by_species(trials)
    total_trials = len(trials)
    total_blocks = len(blocks)
    logger.info("Loaded %d trials across %d species", total_trials, total_blocks)

    if dry_run:
        global_idx = 0
        for bi, block in enumerate(blocks, 1):
            code = block["ebird_code"]
            common = block["common_name"]
            ref = SCRIPT_DIR / block["ref_path"]
            ref_ok = "OK" if ref.exists() else "MISSING"
            print(f"\n  Species {bi}/{total_blocks}: {common} ({code})  ref={ref_ok}")
            for t in block["trials"]:
                global_idx += 1
                sp = SCRIPT_DIR / t["sample_path"]
                status = "OK" if sp.exists() else "MISSING"
                print(f"    Trial {global_idx}: {t['system']} — {status}")
        print(f"\nDry run complete. {total_trials} trials, {total_blocks} species.")
        return

    mapping_path = SCRIPT_DIR / "poll_mapping.json"
    poll_mapping = []
    if mapping_path.exists() and resume_from > 1:
        with open(mapping_path) as f:
            poll_mapping = json.load(f)
        logger.info("Resuming: loaded %d existing poll mappings", len(poll_mapping))

    bot = Bot(token=token)

    global_offset = 0
    for bi, block in enumerate(blocks, 1):
        block_first_trial = block["trials"][0]["trial_id"]

        if block_first_trial < resume_from:
            global_offset += len(block["trials"])
            continue

        logger.info(
            "Posting species %d/%d: %s ...",
            bi,
            total_blocks,
            block["ebird_code"],
        )

        entries = await post_species_block(
            bot,
            channel_id,
            block,
            bi,
            total_blocks,
            global_offset,
            total_trials,
            delay,
        )
        poll_mapping.extend(entries)

        with open(mapping_path, "w") as f:
            json.dump(poll_mapping, f, indent=2)

        global_offset += len(block["trials"])

    logger.info("All %d trials posted. Poll mapping: %s", total_trials, mapping_path)


# ---------------------------------------------------------------------------
# Collect poll results
# ---------------------------------------------------------------------------


async def collect_results(token: str, channel_id: str):
    mapping_path = SCRIPT_DIR / "poll_mapping.json"
    if not mapping_path.exists():
        raise SystemExit(f"No poll mapping found at {mapping_path}")

    with open(mapping_path) as f:
        poll_mapping = json.load(f)

    bot = Bot(token=token)
    results = []

    for entry in poll_mapping:
        try:
            poll = await bot.stop_poll(
                chat_id=channel_id,
                message_id=entry["message_id"],
            )
        except Exception as e:
            logger.warning(
                "Could not stop poll for trial %d (msg %d): %s",
                entry["trial_id"],
                entry["message_id"],
                e,
            )
            continue

        option_votes = [opt.voter_count for opt in poll.options]
        total_votes = sum(option_votes)
        if total_votes == 0:
            logger.warning("No votes for trial %d", entry["trial_id"])
            continue

        vote_distribution = {}
        for opt_idx, count in enumerate(option_votes):
            mos_val = POLL_VALUE_MAP[opt_idx]
            vote_distribution[mos_val] = count

        weighted_sum = sum(v * c for v, c in vote_distribution.items())
        mean_mos = weighted_sum / total_votes

        results.append(
            {
                "trial_id": entry["trial_id"],
                "mos": round(mean_mos, 2),
                "total_votes": total_votes,
                "vote_distribution": vote_distribution,
            }
        )
        logger.info(
            "Trial %d (%s): mean MOS = %.2f (%d votes)",
            entry["trial_id"],
            entry.get("system", "?"),
            mean_mos,
            total_votes,
        )

    out_path = SCRIPT_DIR / "mos_responses_channel.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved %d trial results to %s", len(results), out_path)

    simple_path = SCRIPT_DIR / "mos_responses.json"
    simple = [{"trial_id": r["trial_id"], "mos": r["mos"]} for r in results]
    with open(simple_path, "w") as f:
        json.dump(simple, f, indent=2)
    logger.info("Saved analyze-compatible file to %s", simple_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Post MOS listening test to a Telegram channel"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("TELEGRAM_BOT_TOKEN"),
        help="Bot token (or TELEGRAM_BOT_TOKEN env var)",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default=os.environ.get("TELEGRAM_CHANNEL_ID"),
        help="Channel ID (or TELEGRAM_CHANNEL_ID env var)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=SCRIPT_DIR / "survey_order.csv",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds between posts (avoid rate limits)",
    )
    parser.add_argument(
        "--resume-from",
        type=int,
        default=1,
        help="Resume from trial N (1-based, skips earlier species blocks)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify audio files without posting",
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Stop polls and collect results",
    )
    args = parser.parse_args()

    if not args.token:
        raise SystemExit("No token. Set TELEGRAM_BOT_TOKEN or use --token.")
    if not args.channel:
        raise SystemExit("No channel. Set TELEGRAM_CHANNEL_ID or use --channel.")

    if args.collect:
        asyncio.run(collect_results(args.token, args.channel))
    else:
        asyncio.run(
            post_all(
                args.token,
                args.channel,
                args.csv,
                args.delay,
                args.resume_from,
                args.dry_run,
            )
        )


if __name__ == "__main__":
    main()
