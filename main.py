from debate import DebateSystem
import os
from debate import logger

def main():
    """Main function to run the debate system"""
    # Get API key from environment or user input
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        api_key = input("Enter your Google API key: ").strip()
    
    try:
        debate_system = DebateSystem(api_key)
        final_state = debate_system.run_debate()
        
        print("\n" + "="*50)
        print("DEBATE COMPLETE!")
        print("="*50)
        print(f"Winner: {final_state['winner']}")
        print(f"Check 'debate_log.txt' and 'complete_debate_log.json' for full logs.")
        
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"System error: {e}")

if __name__ == "__main__":
    main()