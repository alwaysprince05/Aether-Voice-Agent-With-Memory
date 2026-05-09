#!/usr/bin/env python3
"""Voice AI Agent demo application."""

import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Voice AI Agent with Memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py                  # Voice mode (microphone + speakers)
  python demo.py --text-mode      # Text mode (keyboard + terminal)
        """
    )
    parser.add_argument(
        "--text-mode",
        action="store_true",
        help="Use text input/output instead of voice"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Import here so config errors surface cleanly
    try:
        from src.voice_agent import VoiceAgent
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please set GROQ_API_KEY in your .env file.")
        sys.exit(1)
    
    agent = VoiceAgent(text_mode=args.text_mode)
    
    mode = "text" if args.text_mode else "voice"
    print(f"Voice AI Agent started in {mode} mode.")
    if args.text_mode:
        print("Type your message and press Enter. Type 'quit' to exit.")
    else:
        print("Speak to interact. Say 'quit' or press Ctrl+C to exit.")
    print()
    
    agent.run()

if __name__ == "__main__":
    main()
