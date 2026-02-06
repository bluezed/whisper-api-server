import argparse
from app import WhisperServiceAPI

def main():
    """
    Local, OpenAI-compatible speech recognition API service using Whisper model.
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Speech recognition service using Whisper model")
    parser.add_argument("--config", help="Path to configuration file", default="config.json")
    
    args = parser.parse_args()
    
    # Start service
    service = WhisperServiceAPI(args.config)
    service.run()


if __name__ == "__main__":
    main()
