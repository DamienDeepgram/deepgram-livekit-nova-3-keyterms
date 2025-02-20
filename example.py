import os
import asyncio
from deepgram import Deepgram
from pathlib import Path

def load_env():
    env_path = Path(__file__).parents[2] / '.env.local'
    if not env_path.exists():
        raise FileNotFoundError(".env.local file not found in project root")
    
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

async def main():
    # Load environment variables
    load_env()
    
    # Initialize the Deepgram client
    DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY')
    if not DEEPGRAM_API_KEY:
        raise ValueError("DEEPGRAM_API_KEY not found in .env.local file")
    
    dg_client = Deepgram(DEEPGRAM_API_KEY)
    
    # Path to your audio file
    AUDIO_FILE_PATH = "test.wav"  # You'll need to provide this file
    
    # Check if audio file exists
    if not os.path.exists(AUDIO_FILE_PATH):
        raise FileNotFoundError(f"Audio file not found at {AUDIO_FILE_PATH}")
    
    print("Testing Deepgram Nova-3 model with keyterms...")
    
    try:
        with open(AUDIO_FILE_PATH, 'rb') as audio:
            source = {'buffer': audio, 'mimetype': 'audio/wav'}
            options = {
                'model': 'nova-3',
                'smart_format': True,
                'language': 'en',
                'keyterm': ['test', 'hello']  # Example keyterms matching our test audio
            }
            
            response = await dg_client.transcription.prerecorded(source, options)
            
            # Print the transcription results
            print("\nTranscription Results:")
            print("-" * 50)
            if 'results' in response and 'channels' in response['results']:
                for channel in response['results']['channels']:
                    if 'alternatives' in channel and len(channel['alternatives']) > 0:
                        transcript = channel['alternatives'][0]['transcript']
                        print(f"Transcript: {transcript}")
                        
                        # Print keyterm detection results if available
                        if 'keyterm' in channel['alternatives'][0]:
                            print("\nKeyterm Detections:")
                            print("-" * 50)
                            keyterms = channel['alternatives'][0]['keyterm']
                            for term, occurrences in keyterms.items():
                                print(f"Term '{term}' detected {len(occurrences)} times:")
                                for occurrence in occurrences:
                                    print(f"  - Confidence: {occurrence.get('confidence', 'N/A')}")
                                    print(f"    Start: {occurrence.get('start', 'N/A')}s")
                                    print(f"    End: {occurrence.get('end', 'N/A')}s")
            else:
                print("No transcription results found in the response")
            
            # Print some metadata about the response
            print("\nMetadata:")
            print("-" * 50)
            if 'metadata' in response:
                print(f"Model: {response['metadata'].get('model', 'N/A')}")
                print(f"Duration: {response['metadata'].get('duration', 'N/A')} seconds")
                print(f"Request ID: {response['metadata'].get('request_id', 'N/A')}")
            
            # Print the full options sent to the API
            print("\nAPI Options Sent:")
            print("-" * 50)
            print(options)
            
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
    
    print("\nTest completed.")

if __name__ == "__main__":
    asyncio.run(main()) 
