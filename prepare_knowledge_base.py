#!/usr/bin/env python3
"""Download, clean, and anonymize a small fictional knowledge base."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests


API_URL = "https://en.wikipedia.org/w/api.php"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
SOURCE_TITLES = (
    "Luke Skywalker",
    "Darth Vader",
    "Princess Leia",
    "Han Solo",
    "Chewbacca",
    "Obi-Wan Kenobi",
    "Yoda",
    "Palpatine",
    "Grand Moff Tarkin",
    "Padmé Amidala",
    "C-3PO",
    "R2-D2",
    "Boba Fett",
    "Jabba the Hutt",
    "Lando Calrissian",
    "Mace Windu",
    "Qui-Gon Jinn",
    "Darth Maul",
    "Count Dooku",
    "Ahsoka Tano",
    "Kylo Ren",
    "Rey (Star Wars)",
    "Finn (Star Wars)",
    "Poe Dameron",
    "Grogu",
    "The Mandalorian (character)",
    "Jedi",
    "Sith",
    "The Force",
    "Galactic Empire (Star Wars)",
    "Rebel Alliance",
    "Death Star",
    "Millennium Falcon",
    "TIE fighter",
    "X-wing fighter",
    "Lightsaber",
    "Stormtrooper (Star Wars)",
    "Clone trooper",
)

# Longer and more specific names are replaced before generic terms.
TERM_MAP = {
    "Alliance to Restore the Republic": "Free Systems Coalition",
    "The Mandalorian": "The Argent Wanderer",
    "Galactic Empire": "Obsidian Directorate",
    "Galactic Republic": "Stellar Concord",
    "Anakin Skywalker": "Aeron Ardyn",
    "Luke Skywalker": "Kael Ardyn",
    "Rey Skywalker": "Rhea Ardyn",
    "Darth Sidious": "Sareth Mor",
    "Darth Vader": "Xarn Velgor",
    "Princess Leia": "Lyra Voss",
    "Leia Organa": "Lyra Voss",
    "Obi-Wan Kenobi": "Orin Valis",
    "Padmé Amidala": "Selene Ardyn",
    "Jabba the Hutt": "Gorba the Vast",
    "Lando Calrissian": "Dalen Kors",
    "Mace Windu": "Taren Vos",
    "Qui-Gon Jinn": "Quorin Jaal",
    "Count Dooku": "Count Varos",
    "Ahsoka Tano": "Ashara Teln",
    "Poe Dameron": "Phael Damar",
    "Boba Fett": "Varek Tann",
    "Kylo Ren": "Kyren Vale",
    "Ben Solo": "Ben Calder",
    "Han Solo": "Renn Calder",
    "Chewbacca": "Bronn Kesh",
    "Darth Maul": "Zorath Kain",
    "Grand Moff Tarkin": "High Prefect Valcor",
    "Palpatine": "Sareth Mor",
    "Snoke": "Othren",
    "Grogu": "Nym",
    "C-3PO": "CX-9",
    "R2-D2": "RT-7",
    "BB-8": "BX-4",
    "Star Wars": "Chronicles of the Aether",
    "Skywalker saga": "Ardyn Cycle",
    "The Empire Strikes Back": "Shadows Ascendant",
    "Return of the Jedi": "Return of the Wardens",
    "The Phantom Menace": "Veiled Omen",
    "Attack of the Clones": "Rise of Echoes",
    "Revenge of the Sith": "Fall of the Covenant",
    "The Force Awakens": "The Current Awakens",
    "The Last Jedi": "The Last Warden",
    "The Rise of Skywalker": "The Rise of Ardyn",
    "A New Hope": "The First Dawn",
    "Rogue One": "Vanguard One",
    "Rebel Alliance": "Free Systems Coalition",
    "New Republic": "Renewed Concord",
    "First Order": "Ascendant Order",
    "Clone Wars": "Replicant Wars",
    "Death Star": "Void Core",
    "Death Stars": "Void Cores",
    "Millennium Falcon": "Dawn Skipper",
    "TIE fighter": "Shade fighter",
    "X-wing fighter": "Talon-wing fighter",
    "clone trooper": "echo trooper",
    "stormtrooper": "voidtrooper",
    "lightsaber": "arcblade",
    "Mandalorian": "Argentian",
    "Jedi Order": "Aether Warden Order",
    "Jedi": "Aether Warden",
    "Sith Order": "Null Covenant",
    "Sith": "Nullbound",
    "the Force": "the Aether Current",
    "Coruscant": "Aurinox",
    "Tatooine": "Draxos",
    "Alderaan": "Elaris",
    "Naboo": "Navira",
    "Dagobah": "Murkworld",
    "Mustafar": "Pyraxis",
    "Kashyyyk": "Varduun",
    "Kamino": "Pelagos",
    "Endor": "Eryndor",
    "Hoth": "Nivora",
    "George Lucas": "Ilyan Voss",
    "Lucasfilm": "Meridian Archives",
    "Skywalker": "Ardyn",
    "Vader": "Velgor",
    "Sidious": "Sareth",
    "Anakin": "Aeron",
    "Luke": "Kael",
    "Leia": "Lyra",
    "Organa": "Voss",
    "Han": "Renn",
    "Solo": "Calder",
    "Chewie": "Bronn",
    "Obi-Wan": "Orin",
    "Kenobi": "Valis",
    "Padmé": "Selene",
    "Amidala": "Ardyn",
    "Boba": "Varek",
    "Fett": "Tann",
    "Jabba": "Gorba",
    "Hutt": "Ghoran",
    "Lando": "Dalen",
    "Calrissian": "Kors",
    "Mace": "Taren",
    "Windu": "Vos",
    "Qui-Gon": "Quorin",
    "Jinn": "Jaal",
    "Maul": "Kain",
    "Dooku": "Varos",
    "Ahsoka": "Ashara",
    "Tano": "Teln",
    "Kylo": "Kyren",
    "Dameron": "Damar",
    "Tarkin": "Valcor",
    "Mandalore": "Argentum",
    "Wookiee": "Vardani",
    "Tusken Raiders": "Dune Clans",
    "Mos Eisley": "Dusthaven",
    "Battle of Yavin": "Battle of Meridian",
    "Cloud City": "Zephyr Spire",
    "Confederacy of Independent Systems": "League of Sovereign Worlds",
    "Jedi Council": "Aether Council",
    "Order 66": "Directive Ashfall",
    "stormtroopers": "voidtroopers",
    "clone troopers": "echo troopers",
    "lightsabers": "arcblades",
    "X-wings": "Talon-wings",
    "X-wing": "Talon-wing",
    "TIE": "Shade",
    "Rebels": "Freesiders",
    "Imperial": "Directorate",
    "Separatists": "Secessionists",
    "Ewoks": "Sylvans",
    "Ewok": "Sylvan",
    "Gungans": "Maruvians",
    "Gungan": "Maruvian",
    "Twi'lek": "Velari",
    "droids": "automatons",
    "droid": "automaton",
    "blasters": "pulse guns",
    "blaster": "pulse gun",
    "hyperdrive": "slipstream drive",
    "Bespin": "Caelora",
    "Geonosis": "Kharaxis",
    "Jakku": "Ravaryn",
    "Ahch-To": "Isle Veyra",
    "Crait": "Salaris",
    "Yavin": "Meridian",
    "Falcon": "Skipper",
    "Trade Federation": "Mercantile Compact",
    "Jar Jar Binks": "Pell Arvo",
    "AT-AT": "Titan walker",
    "Sarlacc": "pit maw",
    "rancor": "dreadbeast",
    "wampa": "ice prowler",
    "carbonite": "stasis crystal",
    "Padawan": "Aether initiate",
    "Emperor": "Sovereign",
    "Rebel": "Freesider",
    "the Empire": "the Directorate",
    "the Republic": "the Concord",
    "the Resistance": "the Ember Guard",
    "the Rebellion": "the Uprising",
    "the Alliance": "the Coalition",
    "the dark side": "the Umbral Path",
    "the light side": "the Luminous Path",
    "Force": "Aether Current",
    "Rey": "Rhea",
    "Finn": "Korr",
    "Yoda": "Master Vael",
}

STOP_SECTIONS = {
    "see also",
    "notes",
    "references",
    "sources",
    "external links",
    "further reading",
}

REAL_WORLD_MARKERS = (
    "20th century",
    " according to",
    "adam driver",
    "alec guinness",
    "anthony daniels",
    " actor",
    " actress",
    " animation",
    " artist",
    " audition",
    " author",
    " award",
    "billy dee williams",
    "carrie fisher",
    "christopher lee",
    "clint eastwood",
    " comic",
    " concept",
    " created by",
    " creator",
    " critics",
    "daisy ridley",
    "dave filoni",
    "david prowse",
    " debut",
    " designer",
    " described ",
    "disney",
    " director",
    " dvd",
    " episode",
    "ewan mcgregor",
    " film",
    " filming",
    " filmmaker",
    "frank oz",
    " franchise",
    "george lucas",
    "harrison ford",
    "hayden christensen",
    "ian mcdiarmid",
    "industrial light",
    "j. j. abrams",
    "j. abrams",
    "james earl jones",
    "jeremy bulloch",
    "john boyega",
    "jon favreau",
    "kenny baker",
    "liam neeson",
    "lucas stated",
    "lucas",
    "mark hamill",
    "hamill",
    " merchandise",
    "michael arndt",
    "natalie portman",
    " novel",
    " nominated",
    "oscar isaac",
    "pedro pascal",
    "peter mayhew",
    " portray",
    " production",
    " puppeteer",
    " ranked ",
    "ray park",
    "rolling stone",
    " reception",
    " release",
    " screen rant",
    "samuel l. jackson",
    " screenplay",
    " screenwriter",
    " script",
    " series",
    "sharon tate",
    "squeaky fromme",
    "temuera morrison",
    "the daily beast",
    "the telegraph",
    " television",
    " trilogy",
    " video game",
    " voice",
    " was cast",
    " was conceived",
    " was designed",
    " william katt",
    " writer",
    "donald glover",
    "entertainment weekly",
    "guinness",
    " interview",
    "neeson",
    "prowse",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download English Wikipedia articles, clean their text, replace key "
            "terms, and create an anonymized Markdown knowledge base."
        ),
        epilog=(
            "Example: ./prepare_knowledge_base.py --output-dir knowledge_base "
            "--min-documents 30"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("knowledge_base"),
        help="Output directory (default: knowledge_base).",
    )
    parser.add_argument(
        "--min-documents",
        type=int,
        default=30,
        help="Minimum number of documents required for success (default: 30).",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=120,
        help="Skip source pages shorter than this after cleaning (default: 120).",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=900,
        help="Maximum words retained from each source page (default: 900).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds (default: 30).",
    )
    return parser.parse_args()


def compile_term_pattern(term_map: dict[str, str]) -> re.Pattern[str]:
    terms = sorted(term_map, key=len, reverse=True)
    alternatives = "|".join(re.escape(term) for term in terms)
    return re.compile(rf"(?<!\w)({alternatives})(?!\w)", re.IGNORECASE)


def replace_terms(text: str, term_map: dict[str, str], pattern: re.Pattern[str]) -> str:
    replacements = {key.casefold(): value for key, value in term_map.items()}
    return pattern.sub(lambda match: replacements[match.group(0).casefold()], text)


def clean_extract(extract: str, max_words: int) -> str:
    paragraphs: list[str] = []
    for raw_line in extract.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        heading = re.fullmatch(r"=+\s*(.*?)\s*=+", line)
        if heading:
            if heading.group(1).casefold() in STOP_SECTIONS:
                break
            continue
        line = re.sub(r"\s+", " ", line)
        paragraphs.append(line)

    sentences = re.split(r"(?<=[.!?])\s+", " ".join(paragraphs))
    fictional_sentences = [
        sentence
        for sentence in sentences
        if len(sentence.split()) >= 4
        and not any(marker in sentence.casefold() for marker in REAL_WORLD_MARKERS)
    ]
    words = " ".join(fictional_sentences).split()
    return " ".join(words[:max_words]).strip()


def fetch_extract(session: requests.Session, title: str, timeout: float) -> str:
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "prop": "extracts",
        "explaintext": "1",
        "redirects": "1",
        "titles": title,
    }
    response = session.get(API_URL, params=params, timeout=timeout)
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        raise ValueError("page was not found")
    extract = pages[0].get("extract", "")
    if not extract.strip():
        raise ValueError("page has no plain-text extract")
    return extract


def source_url(title: str) -> str:
    return f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"


def write_sources(path: Path, sources: list[tuple[str, str, str]]) -> None:
    lines = [
        "# Источники базы знаний",
        "",
        (
            "Тексты получены из англоязычной Wikipedia, сокращены, очищены и "
            "преобразованы заменой ключевых терминов. Изменения выполнены для "
            "учебной проверки RAG."
        ),
        "",
        (
            "Материалы распространяются на условиях "
            f"[CC BY-SA 4.0]({LICENSE_URL})."
        ),
        "",
        "| Source ID | Исходная статья | Документ |",
        "|---|---|---|",
    ]
    for source_id, title, filename in sources:
        lines.append(f"| {source_id} | [{title}]({source_url(title)}) | `{filename}` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_args(args: argparse.Namespace) -> None:
    if args.min_documents < 1:
        raise ValueError("--min-documents must be positive")
    if args.min_words < 1:
        raise ValueError("--min-words must be positive")
    if args.max_words < args.min_words:
        raise ValueError("--max-words must be greater than or equal to --min-words")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")


def main() -> int:
    args = parse_args()
    print("Knowledge-base preparation started")
    print(f"Output directory: {args.output_dir.resolve()}")
    print(f"Required documents: {args.min_documents}")
    try:
        validate_args(args)
    except ValueError as error:
        print(f"Invalid arguments: {error}")
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created or reused directory: {args.output_dir}")
    term_pattern = compile_term_pattern(TERM_MAP)
    session = requests.Session()
    session.headers["User-Agent"] = (
        "architecture-pro-rag-study/1.0 "
        "(educational project; https://github.com/)"
    )

    sources: list[tuple[str, str, str]] = []
    failures: list[str] = []
    started = time.monotonic()

    for title in SOURCE_TITLES:
        print(f"Downloading source: {title}")
        try:
            extract = fetch_extract(session, title, args.timeout)
            cleaned = clean_extract(extract, args.max_words)
            if len(cleaned.split()) < args.min_words:
                raise ValueError(
                    f"only {len(cleaned.split())} words after cleaning; "
                    f"minimum is {args.min_words}"
                )
            transformed_title = replace_terms(title, TERM_MAP, term_pattern)
            transformed_text = replace_terms(cleaned, TERM_MAP, term_pattern)
            if term_pattern.search(f"{transformed_title}\n{transformed_text}"):
                raise ValueError("an original key term remained after replacement")

            sequence = len(sources) + 1
            source_id = f"SRC-{sequence:03d}"
            filename = f"entity_{sequence:02d}.md"
            output_path = args.output_dir / filename
            content = f"# {transformed_title}\n\n{transformed_text}\n"
            output_path.write_text(content, encoding="utf-8")
            print(f"Created document: {output_path}")
            sources.append((source_id, title, filename))
        except (requests.RequestException, ValueError, KeyError) as error:
            failures.append(title)
            print(f"Failed source {title}: {error}")

    terms_path = args.output_dir / "terms_map.json"
    terms_path.write_text(
        json.dumps(TERM_MAP, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Created term map: {terms_path}")

    sources_path = args.output_dir / "SOURCES.md"
    write_sources(sources_path, sources)
    print(f"Created source manifest: {sources_path}")

    elapsed = time.monotonic() - started
    print(f"Documents created: {len(sources)}")
    print(f"Failed sources: {len(failures)}")
    if failures:
        print(f"Failed source titles: {', '.join(failures)}")
    print(f"Elapsed seconds: {elapsed:.2f}")

    if len(sources) < args.min_documents or failures:
        print("Knowledge-base preparation finished with errors")
        return 1
    print("Knowledge-base preparation finished successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
