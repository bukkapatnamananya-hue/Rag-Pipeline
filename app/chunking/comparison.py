import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


CHUNK_DIR = Path("data/chunks")
OUTPUT_DIR = Path("data/comparison")


FILES = {
    "Recursive": CHUNK_DIR / "recursive_chunks.json",
    "Structure-Aware": CHUNK_DIR / "structure_aware_chunks.json",
    "Contextual": CHUNK_DIR / "contextual_chunks.json",
}


def load_chunks(path):
    """Load chunks from a JSON file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {path.resolve()}"
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def calculate_statistics(chunks):
    """Calculate basic statistics for a chunking strategy."""

    character_counts = [
        chunk["character_count"]
        for chunk in chunks
    ]

    word_counts = [
        chunk["word_count"]
        for chunk in chunks
    ]

    sections = set()

    for chunk in chunks:

        section = chunk.get("section")

        if section:
            sections.add(section)

    return {
        "total_chunks": len(chunks),

        "average_characters": round(
            mean(character_counts), 2
        ),

        "minimum_characters": min(
            character_counts
        ),

        "maximum_characters": max(
            character_counts
        ),

        "average_words": round(
            mean(word_counts), 2
        ),

        "minimum_words": min(
            word_counts
        ),

        "maximum_words": max(
            word_counts
        ),

        "number_of_sections": len(sections),
    }


def print_comparison(results):
    """Print comparison table."""

    print("\n")
    print("=" * 95)
    print("CHUNKING STRATEGY COMPARISON")
    print("=" * 95)

    print(
        f"{'Strategy':<20}"
        f"{'Chunks':>10}"
        f"{'Avg Chars':>15}"
        f"{'Min Chars':>15}"
        f"{'Max Chars':>15}"
        f"{'Avg Words':>15}"
    )

    print("-" * 95)

    for strategy, stats in results.items():

        print(
            f"{strategy:<20}"
            f"{stats['total_chunks']:>10}"
            f"{stats['average_characters']:>15}"
            f"{stats['minimum_characters']:>15}"
            f"{stats['maximum_characters']:>15}"
            f"{stats['average_words']:>15}"
        )

    print("=" * 95)

    print("\nNumber of sections represented:")

    for strategy, stats in results.items():

        print(
            f"  {strategy}: "
            f"{stats['number_of_sections']}"
        )


def save_statistics(results):
    """Save comparison statistics as JSON."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR / "chunk_statistics.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=2
        )

    print(
        f"\nStatistics saved to: {output_path}"
    )


def create_chunk_count_chart(results):
    """Create chart comparing total chunk counts."""

    strategies = list(results.keys())

    counts = [
        results[strategy]["total_chunks"]
        for strategy in strategies
    ]

    plt.figure(figsize=(8, 5))

    plt.bar(
        strategies,
        counts
    )

    plt.title(
        "Number of Chunks by Strategy"
    )

    plt.xlabel(
        "Chunking Strategy"
    )

    plt.ylabel(
        "Number of Chunks"
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR / "chunk_count_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved chart: {output_path}"
    )


def create_average_size_chart(results):
    """Create chart comparing average chunk size."""

    strategies = list(results.keys())

    averages = [
        results[strategy]["average_characters"]
        for strategy in strategies
    ]

    plt.figure(figsize=(8, 5))

    plt.bar(
        strategies,
        averages
    )

    plt.title(
        "Average Chunk Size by Strategy"
    )

    plt.xlabel(
        "Chunking Strategy"
    )

    plt.ylabel(
        "Average Characters"
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR / "average_chunk_size_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved chart: {output_path}"
    )


def create_size_distribution(chunks_by_strategy):
    """Create a chunk-size distribution chart for each strategy."""

    filename_map = {
        "Recursive": "recursive_size_distribution.png",
        "Structure-Aware": "structure_aware_size_distribution.png",
        "Contextual": "contextual_size_distribution.png",
    }

    for strategy, chunks in chunks_by_strategy.items():

        sizes = [
            chunk["character_count"]
            for chunk in chunks
        ]

        plt.figure(figsize=(8, 5))

        plt.hist(
            sizes,
            bins=10
        )

        plt.title(
            f"Chunk Size Distribution - {strategy}"
        )

        plt.xlabel(
            "Characters per Chunk"
        )

        plt.ylabel(
            "Number of Chunks"
        )

        plt.tight_layout()

        filename = filename_map[strategy]

        output_path = (
            OUTPUT_DIR / filename
        )

        plt.savefig(
            output_path,
            dpi=300
        )

        plt.close()

        print(
            f"Saved chart: {output_path}"
        )


def main():

    print("=" * 60)
    print("CHUNKING COMPARISON")
    print("=" * 60)

    chunks_by_strategy = {}

    results = {}

    # --------------------------------------------------------
    # Load all three strategies
    # --------------------------------------------------------

    for strategy, path in FILES.items():

        print(
            f"\nLoading {strategy}: {path}"
        )

        chunks = load_chunks(path)

        chunks_by_strategy[strategy] = chunks

        results[strategy] = calculate_statistics(
            chunks
        )

        print(
            f"Loaded {len(chunks)} chunks"
        )

    # --------------------------------------------------------
    # Print statistics
    # --------------------------------------------------------

    print_comparison(results)

    # --------------------------------------------------------
    # Save statistics
    # --------------------------------------------------------

    save_statistics(results)

    # --------------------------------------------------------
    # Create visualizations
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nCreating visualizations...")

    create_chunk_count_chart(
        results
    )

    create_average_size_chart(
        results
    )



    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()