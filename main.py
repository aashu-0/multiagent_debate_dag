#!/usr/bin/env python3
"""
LangGraph Debate Simulation System
A structured debate system where two AI agents engage in argumentation
with memory management, turn control, and automated judging.
"""

import json
import logging
from datetime import datetime
from typing import Literal, TypedDict, List, Dict, Any
from dataclasses import dataclass

# LangGraph imports
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

# Google Gemini imports
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debate_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DebateMessage:
    """Structured debate message"""
    speaker: str
    content: str
    round_num: int
    timestamp: str

class DebateState(TypedDict):
    """Enhanced state for debate management"""
    topic: str
    messages: List[DebateMessage]
    current_round: int
    current_speaker: str
    debate_summary: str
    winner: str
    winner_reason: str
    is_complete: bool

class DebateSystem:
    """Main debate system using LangGraph"""
    
    def __init__(self, api_key: str):
        """Initialize the debate system with Gemini API"""
        genai.configure(api_key=api_key)
        
        # Configure Gemini model
        self.model = genai.GenerativeModel(
            'gemini-1.5-pro',
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        # Agent personas
        self.agent_personas = {
            "Scientist": """You are a distinguished scientist with expertise in empirical research, 
            data analysis, and evidence-based reasoning. You approach debates with:
            - Emphasis on peer-reviewed research and data
            - Risk assessment and public safety concerns
            - Systematic methodology and logical progression
            - Practical implications and real-world applications
            Keep your arguments concise (2-3 sentences) and grounded in scientific principles.""",
            
            "Philosopher": """You are a renowned philosopher specializing in ethics, logic, and 
            theoretical frameworks. You approach debates with:
            - Deep questioning of assumptions and premises
            - Ethical implications and moral reasoning
            - Historical context and theoretical perspectives
            - Abstract thinking and conceptual analysis
            Keep your arguments concise (2-3 sentences) and philosophically rigorous."""
        }
        
        # Initialize memory and graph
        self.memory = MemorySaver()
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        graph_builder = StateGraph(DebateState)
        
        # Add nodes
        graph_builder.add_node("user_input", self.user_input_node)
        graph_builder.add_node("scientist", self.scientist_node)
        graph_builder.add_node("philosopher", self.philosopher_node)
        graph_builder.add_node("supervisor", self.supervisor_node)
        graph_builder.add_node("judge", self.judge_node)
        graph_builder.add_node("memory_update", self.memory_update_node)
        
        # Define edges
        graph_builder.add_edge(START, "user_input")
        graph_builder.add_edge("user_input", "supervisor")
        graph_builder.add_edge("scientist", "memory_update")
        graph_builder.add_edge("philosopher", "memory_update")
        graph_builder.add_edge("memory_update", "supervisor")
        graph_builder.add_edge("judge", END)
        
        return graph_builder.compile(checkpointer=self.memory)
    
    def user_input_node(self, state: DebateState) -> DebateState:
        """Accept debate topic from user"""
        if not state.get("topic"):
            topic = input("\nEnter topic for debate: ").strip()
            logger.info(f"Debate topic set: {topic}")
            
            return {
                **state,
                "topic": topic,
                "messages": [],
                "current_round": 1,
                "current_speaker": "Scientist",
                "debate_summary": "",
                "winner": "",
                "winner_reason": "",
                "is_complete": False
            }
        return state
    
    def scientist_node(self, state: DebateState) -> Command[Literal["memory_update"]]:
        """Scientist agent node"""
        return self._agent_response(state, "Scientist")
    
    def philosopher_node(self, state: DebateState) -> Command[Literal["memory_update"]]:
        """Philosopher agent node"""
        return self._agent_response(state, "Philosopher")
    
    def _agent_response(self, state: DebateState, agent_name: str) -> Command[Literal["memory_update"]]:
        """Generic agent response handler"""
        # Get recent context (last 3 messages)
        recent_messages = state["messages"][-3:] if len(state["messages"]) > 3 else state["messages"]
        context = "\n".join([f"{msg.speaker}: {msg.content}" for msg in recent_messages])
        
        prompt = f"""
        {self.agent_personas[agent_name]}
        
        DEBATE TOPIC: {state['topic']}
        
        CURRENT ROUND: {state['current_round']}/8
        
        RECENT CONTEXT:
        {context}
        
        Your task: Provide a compelling argument for your perspective on this topic.
        Requirements:
        - Keep response to 2-3 sentences maximum
        - Make a distinct point (don't repeat previous arguments)
        - Be persuasive and logical
        - Stay in character as a {agent_name.lower()}
        
        Your argument:
        """
        
        try:
            response = self.model.generate_content(prompt)
            argument = response.text.strip()
            
            # Create message
            message = DebateMessage(
                speaker=agent_name,
                content=argument,
                round_num=state["current_round"],
                timestamp=datetime.now().isoformat()
            )
            
            # Log the argument
            print(f"\n[Round {state['current_round']}] {agent_name}: {argument}")
            logger.info(f"Round {state['current_round']} - {agent_name}: {argument}")
            
            return Command(
                goto="memory_update",
                update={"messages": state["messages"] + [message]}
            )
            
        except Exception as e:
            logger.error(f"Error in {agent_name} node: {e}")
            # Fallback response
            fallback_message = DebateMessage(
                speaker=agent_name,
                content=f"I maintain my position on {state['topic']} based on {agent_name.lower()} principles.",
                round_num=state["current_round"],
                timestamp=datetime.now().isoformat()
            )
            return Command(
                goto="memory_update",
                update={"messages": state["messages"] + [fallback_message]}
            )
    
    def memory_update_node(self, state: DebateState) -> Command[Literal["supervisor"]]:
        """Update memory and prepare for next turn"""
        # Update current speaker for next turn
        next_speaker = "Philosopher" if state["current_speaker"] == "Scientist" else "Scientist"
        
        # Check if we need to increment round (after philosopher speaks)
        next_round = state["current_round"]
        if state["current_speaker"] == "Philosopher":
            next_round += 1
        
        return Command(
            goto="supervisor",
            update={
                "current_speaker": next_speaker,
                "current_round": next_round
            }
        )
    
    def supervisor_node(self, state: DebateState) -> Command[Literal["scientist", "philosopher", "judge"]]:
        """Supervisor to control debate flow"""
        # Check if debate is complete (8 rounds = 16 total messages)
        if len(state["messages"]) >= 16:
            logger.info("Debate complete, moving to judge")
            return Command(goto="judge")
        
        # Route to appropriate agent
        if state["current_speaker"] == "Scientist":
            return Command(goto="scientist")
        else:
            return Command(goto="philosopher")
    
    def judge_node(self, state: DebateState) -> DebateState:
        """Judge node to evaluate debate and declare winner"""
        # Prepare debate transcript
        transcript = "\n".join([
            f"[Round {msg.round_num}] {msg.speaker}: {msg.content}"
            for msg in state["messages"]
        ])
        
        judge_prompt = f"""
        You are an expert debate judge. Analyze this debate transcript and determine the winner.
        
        TOPIC: {state['topic']}
        
        DEBATE TRANSCRIPT:
        {transcript}
        
        Evaluation Criteria:
        1. Logical consistency and coherence
        2. Strength of evidence and reasoning
        3. Persuasiveness of arguments
        4. Addressing counterarguments
        5. Overall debate strategy
        
        Provide:
        1. A brief summary of the debate (2-3 sentences)
        2. Winner (either "Scientist" or "Philosopher")
        3. Detailed reason for your decision (2-3 sentences)
        
        Format your response as:
        SUMMARY: [your summary]
        WINNER: [Scientist or Philosopher]
        REASON: [your reasoning]
        """
        
        try:
            response = self.model.generate_content(judge_prompt)
            judgment = response.text.strip()
            
            # Parse judgment
            lines = judgment.split('\n')
            summary = ""
            winner = ""
            reason = ""
            
            for line in lines:
                if line.startswith("SUMMARY:"):
                    summary = line.replace("SUMMARY:", "").strip()
                elif line.startswith("WINNER:"):
                    winner = line.replace("WINNER:", "").strip()
                elif line.startswith("REASON:"):
                    reason = line.replace("REASON:", "").strip()
            
            # Display results
            print(f"\n{'='*50}")
            print("DEBATE COMPLETE")
            print(f"{'='*50}")
            print(f"\n[Judge] Summary of debate:\n{summary}")
            print(f"\n[Judge] Winner: {winner}")
            print(f"\nReason: {reason}")
            print(f"\n{'='*50}")
            
            # Log final results
            logger.info(f"Debate Summary: {summary}")
            logger.info(f"Winner: {winner}")
            logger.info(f"Reason: {reason}")
            
            return {
                **state,
                "debate_summary": summary,
                "winner": winner,
                "winner_reason": reason,
                "is_complete": True
            }
            
        except Exception as e:
            logger.error(f"Error in judge node: {e}")
            return {
                **state,
                "debate_summary": "Error in judgment process",
                "winner": "No winner determined",
                "winner_reason": "Technical error occurred",
                "is_complete": True
            }
    
    def generate_dag_diagram(self):
        """Generate and display DAG diagram"""
        try:
            from IPython.display import Image, display
            print("\nGenerating DAG diagram...")
            display(Image(self.graph.get_graph().draw_mermaid_png()))
        except ImportError:
            print("DAG diagram generation requires IPython and graphviz dependencies")
            print("Install with: pip install ipython graphviz")
        except Exception as e:
            print(f"Could not generate DAG diagram: {e}")
            # Print text representation
            print("\nWorkflow Structure:")
            print("START → user_input → supervisor → [scientist|philosopher] → memory_update → supervisor → judge → END")
    
    def run_debate(self):
        """Run the complete debate simulation"""
        print("="*60)
        print("AI DEBATE SIMULATION SYSTEM")
        print("="*60)
        print("Two AI agents (Scientist vs Philosopher) will debate a topic")
        print("The debate consists of 8 rounds (4 arguments per agent)")
        print("="*60)
        
        try:
            # Initialize state
            initial_state = {
                "topic": "",
                "messages": [],
                "current_round": 1,
                "current_speaker": "Scientist",
                "debate_summary": "",
                "winner": "",
                "winner_reason": "",
                "is_complete": False
            }
            
            # Run the graph
            config = {"configurable": {"thread_id": "debate_session"}}
            final_state = self.graph.invoke(initial_state, config)
            
            # Save complete log
            self._save_debate_log(final_state)
            
            return final_state
            
        except KeyboardInterrupt:
            print("\nDebate interrupted by user")
            logger.info("Debate interrupted by user")
        except Exception as e:
            print(f"Error during debate: {e}")
            logger.error(f"Debate error: {e}")
    
    def _save_debate_log(self, final_state: DebateState):
        """Save complete debate log to file"""
        log_data = {
            "topic": final_state["topic"],
            "timestamp": datetime.now().isoformat(),
            "messages": [
                {
                    "speaker": msg.speaker,
                    "content": msg.content,
                    "round": msg.round_num,
                    "timestamp": msg.timestamp
                }
                for msg in final_state["messages"]
            ],
            "summary": final_state["debate_summary"],
            "winner": final_state["winner"],
            "winner_reason": final_state["winner_reason"]
        }
        
        filename = f"debate_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        logger.info(f"Complete debate log saved to {filename}")
        print(f"\nComplete debate log saved to: {filename}")

def main():
    """Main function to run the debate system"""
    print("LangGraph Debate Simulation System")
    print("=================================")
    
    # Get API key
    api_key = input("Enter your Google Gemini API key: ").strip()
    if not api_key:
        print("API key is required!")
        return
    
    try:
        # Initialize and run debate system
        debate_system = DebateSystem(api_key)
        
        # Generate DAG diagram
        debate_system.generate_dag_diagram()
        
        # Run the debate
        final_state = debate_system.run_debate()
        
        if final_state and final_state.get("is_complete"):
            print("\nDebate simulation completed successfully!")
        
    except Exception as e:
        print(f"System error: {e}")
        logger.error(f"System error: {e}")

if __name__ == "__main__":
    main()