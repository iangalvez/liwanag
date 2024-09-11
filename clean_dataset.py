# Imports
import csv
import re
from collections import Counter
from dataclasses import dataclass

# Define the "Pair" class
@dataclass(frozen=True)
class Pair:
    en: str
    en_words: list[str]
    tl: str
    tl_words: list[str]

    def dump(self):
        print(f"\ten={self.en}")
        print(f"\ten_words={self.en_words}")
        print(f"\ttl={self.tl}")
        print(f"\ttl_words={self.tl_words}")

# Split sentences into their constituent words
WORD_BOUNDARY: re.Pattern[str] = re.compile(r"""[\s,\.!?"]""")
def words(line: str) -> list[str]:
    l = [w.strip() for w in re.split(WORD_BOUNDARY, line) if w.strip()]
    # Skip numbers.
    l = [w for w in l if not w.isdigit()]
    return l

# Parse sentence pairs from FILE (in tsv format)
FILE: str = "corpus/Sentence pairs in Tagalog-English - 2024-08-15.tsv"
WORD_LIMIT: int = 15
def parse_sentences() -> list[Pair]:
    pairs: list[Pair] = []
    with open(FILE, "r") as stream:
        reader = csv.reader(stream, delimiter="\t")
        for row in reader:
            en: str = row[3].strip().lower() # make case insensitive!
            tl: str = row[1].strip().lower()
            en_words: list[str] = words(en)
            tl_words: list[str] = words(tl)
            # Skip long sentences.
            if len(tl_words) > WORD_LIMIT:
                continue
            # Skip if the sentence doesn't contain any words.
            if (not en_words) or (not tl_words):
                continue
            pair: Pair = Pair(
                en=en,
                en_words=en_words,
                tl=tl,
                tl_words=tl_words,
            )
            pairs.append(pair)
    print(f"Found {len(pairs):,} sentence pairs.")
    return pairs


# Language frequency tables
def language_frequency_table(sentences: list[list[str]]) -> Counter[str]:
    """
    Given a list of sentences (lists of words), build up a frequency table.
    """
    table: Counter[str] = Counter()
    for sentence in sentences:
        table.update(sentence)
    print(f"\tFound {len(table)} words.")
    first = most_common(table)
    last = least_common(table)
    print(f"\tMost common: '{first}' ({table[first]}).")
    print(f"\tLeast common: '{last}' ({table[last]}).")
    print(f"\tAverage frequency: {counter_avg(table)}")
    return table

# Helpers
def most_common(c: Counter[str]) -> str:
    return c.most_common(1)[0][0]


def least_common(c: Counter[str]) -> str:
    min_frequency = min(c.values())
    least_common_items = [
        item for item, count in c.items() if count == min_frequency
    ]
    return least_common_items[0]


def counter_avg(c: Counter) -> float:
    total = sum(c.values())
    n = len(c)
    average_frequency = total / n
    return average_frequency


# Frequency cutoff
MOST_COMMON_WORDS_CUTOFF: float = 5000
def most_common_words(c: Counter) -> set[str]:
    return set([p[0] for p in c.most_common(MOST_COMMON_WORDS_CUTOFF)])

def sort_pairs(pairs: list[Pair], tl_freq: Counter[str]) -> list[Pair]:
    """
    Sort pairs from shortest and most common Tagalog words. Specifically, we
    sort by the average frequency of the words in the Tagalog sentence, divided
    by the length of the sentence, in reverse order.
    """
    return sorted(
        pairs,
        key=lambda p: avg_freq(p.tl_words, tl_freq) / len(p.tl_words),
        reverse=True,
    )

def avg_freq(words: list[str], tbl: Counter[str]) -> float:
    """
    Return the average frequency for the words.
    """
    return sum(tbl[w] for w in words) / len(words)

# Remove duplicates
def remove_duplicates(pairs: list[Pair]) -> list[Pair]:
    result: list[Pair] = []
    seen_en: set[str] = set()
    seen_tl: set[str] = set()
    skipped: int = 0
    for pair in pairs:
        stripped_en: str = (
            pair.en.replace("!", "").replace(".", "").replace(",", "").strip()
        )
        stripped_tl: str = (
            pair.tl.replace("!", "").replace(".", "").replace(",", "").strip()
        )
        if stripped_en in seen_en:
            skipped += 1
        elif stripped_tl in seen_tl:
            skipped += 1
        else:
            result.append(pair)
            seen_en.add(stripped_en)
            seen_tl.add(stripped_tl)
    print(f"Skipped {skipped} sentence pairs that had the same text.")
    return result

# Build clozes
CLOZE_LIMIT: int = 3

@dataclass(frozen=True)
class Cloze:
    tl: str
    en: str
    clozed_word: str

def minimize(lst, fn):
    """
    Return the value that gives the smallest value of f.
    """
    assert len(lst) > 0
    smallest_index: int = 0
    smallest_value: float = float("inf")
    for (idx, elem) in enumerate(lst):
        val: float = fn(elem)
        if val < smallest_value:
            smallest_index = idx
            smallest_value = val
    return lst[smallest_index]

def build_clozes(
    pairs: list[Pair],
    tl_freq: Counter[str],
    tl_common: set[str],
) -> list[Cloze]:
    clozes: list[Cloze] = []
    # Track how many times we've made a cloze for each word. We don't need too many clozes per word.
    cloze_count_tl: Counter[str] = Counter()
    skipped_limit: int = 0
    skipped_freq: int = 0
    for pair in pairs:
        # Find the rarest words in Tagalog.
        rarest_tl: str = minimize(pair.tl_words, lambda w: tl_freq[w])
        # Cloze the Tagalog word.
        if cloze_count_tl[rarest_tl] == CLOZE_LIMIT:
            skipped_limit += 1
        elif rarest_tl not in tl_common:
            skipped_freq += 1
        else:
            cloze_tl: Cloze = Cloze(
                tl=pair.tl.replace(rarest_tl, "{{c1::" + rarest_tl + "}}"),
                en=pair.en,
                clozed_word=rarest_tl
            )
            clozes.append(cloze_tl)
            cloze_count_tl.update({rarest_tl: 1})
    print(
        f"Skipped {skipped_limit} clozes because the word appeared too many "
        "times."
    )
    print(
        f"Skipped {skipped_freq} clozes because the word was under the "
        "frequency cutoff."
    )
    return clozes

# Dump clozes into units
def dump_clozes(clozes: list[Cloze]):
    print(f"Compiled {len(clozes)} clozes.")
    # Group sentences into units of 100 each.
    units: list[list[Cloze]] = group(clozes, 100)
    print(f"Dumping {len(units)} units.")
    for (unit_id, unit) in enumerate(units):
        with open(f"output/unit_{unit_id}.csv", "w") as stream:
            writer = csv.writer(
                stream,
                delimiter=",",
                quotechar='"',
                quoting=csv.QUOTE_ALL,
                lineterminator="\n",
            )
            for cloze in unit:
                writer.writerow(
                    [f"{cloze.tl.capitalize()} ({cloze.en.capitalize()})", cloze.clozed_word]
                )

def dump_all_clozes(clozes: list[Cloze]):
    print(f"Compiled {len(clozes)} clozes.")
    with open(f"output/all.csv", "w") as stream:
        writer = csv.writer(
            stream,
            delimiter=",",
            quotechar='"',
            quoting=csv.QUOTE_ALL,
            lineterminator="\n",
        )
        for cloze in clozes:
            writer.writerow(
                [f"{cloze.tl.capitalize()} ({cloze.en.capitalize()})", cloze.clozed_word]
            )

# Helper: segment a list into a list of sublists, each sublist containing n items (except for the last)
def group(lst, n):
    result = []
    for i in range(0, len(lst), n):
        result.append(lst[i : i + n])
    return result

def main():
    # Parse sentence pairs.
    pairs: list[Pair] = parse_sentences()
    # Building frequency table.
    print("English frequency table:")
    # en_freq: Counter[str] = language_frequency_table(
    #     [pair.en_words for pair in pairs]
    # )
    print("Tagalog frequency table:")
    tl_freq: Counter[str] = language_frequency_table(
        [pair.tl_words for pair in pairs]
    )
    # Find the frequency cutoff.
    # en_common = most_common_words(en_freq)
    tl_common = most_common_words(tl_freq)
    print("Sorting...")
    pairs = sort_pairs(pairs, tl_freq)
    pairs = remove_duplicates(pairs)
    # Print first and last sentences.
    print("First sentence:")
    pairs[0].dump()
    print("Last sentence:")
    pairs[-1].dump()
    # Build clozes.
    clozes: list[Cloze] = build_clozes(
        pairs, tl_freq, tl_common
    )
    dump_all_clozes(clozes)


if __name__ == "__main__":
    main()