"""
main.py
-------
FitByte AI Content Creator — Command Line Interface

Usage examples:

  # Generate a single blog post (interactive)
  python main.py --topic "why your resting heart rate matters" --channel blog

  # Generate Instagram caption (auto-approve)
  python main.py --topic "winter training" --channel instagram --auto

  # Run a batch of posts
  python main.py --batch

  # Run uniqueness comparison
  python main.py --topic "sleep quality vs quantity" --compare

  # Use Anthropic instead of OpenAI
  python main.py --topic "overtraining signs" --provider anthropic
"""

import argparse
import sys
from pathlib import Path

# Add src to path when running from project root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from content_pipeline import ContentPipeline


CHANNELS = ["blog", "instagram", "linkedin", "email_subject"]
AUDIENCES = ["performance_athlete", "fitness_enthusiast", "health_professional", "upgrader", "general"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="FitByte AI Content Creator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--topic", type=str, help="Content topic or angle")
    parser.add_argument(
        "--channel", type=str, default="blog", choices=CHANNELS,
        help="Target channel (default: blog)"
    )
    parser.add_argument(
        "--audience", type=str, default="fitness_enthusiast", choices=AUDIENCES,
        help="Target audience (default: fitness_enthusiast)"
    )
    parser.add_argument(
        "--provider", type=str, default="auto", choices=["auto", "openai", "anthropic"],
        help="LLM provider (default: auto)"
    )
    parser.add_argument("--auto", action="store_true", help="Skip human review, auto-save")
    parser.add_argument("--compare", action="store_true", help="Run uniqueness comparison")
    parser.add_argument("--batch", action="store_true", help="Run pre-defined batch of posts")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose logging")
    return parser.parse_args()


BATCH_REQUESTS = [
    {
        "topic": "why rest days are part of training",
        "channel": "blog",
        "audience": "fitness_enthusiast",
    },
    {
        "topic": "the difference between sleep length and sleep quality",
        "channel": "blog",
        "audience": "health_professional",
    },
    {
        "topic": "how dual-frequency GPS changes route accuracy",
        "channel": "blog",
        "audience": "performance_athlete",
    },
    {
        "topic": "your body knows you're stressed before your brain does",
        "channel": "instagram",
        "audience": "fitness_enthusiast",
    },
    {
        "topic": "training load and injury prevention",
        "channel": "linkedin",
        "audience": "health_professional",
    },
]


def main():
    args = parse_args()

    print("\n  _____ _ _   ____        _       ")
    print(" |  ___(_) |_| __ ) _   _| |_ ___ ")
    print(" | |_  | | __|  _ \\| | | | __/ _ \\")
    print(" |  _| | | |_| |_) | |_| | ||  __/")
    print(" |_|   |_|\\__|____/ \\__, |\\__\\___|")
    print("                    |___/          ")
    print("  AI Content Creator\n")

    pipeline = ContentPipeline(provider=args.provider, verbose=not args.quiet)

    if args.batch:
        print(f"Running batch: {len(BATCH_REQUESTS)} content pieces\n")
        results = pipeline.run_batch(BATCH_REQUESTS)
        print(f"\n✓ Batch complete. {len(results)} pieces generated in outputs/")
        return

    if not args.topic:
        print("Enter a content topic (or press Ctrl+C to quit):")
        args.topic = input("> ").strip()
        if not args.topic:
            print("No topic provided. Exiting.")
            return

    if args.compare:
        pipeline.compare_uniqueness(topic=args.topic, channel=args.channel)
        return

    result = pipeline.run(
        topic=args.topic,
        channel=args.channel,
        audience=args.audience,
        auto_approve=args.auto,
    )

    if result.get("content") and result["content"] not in ("", "__REGENERATE__"):
        print(f"\n✓ Done. Content saved to outputs/")


if __name__ == "__main__":
    main()
