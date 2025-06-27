# debate_system.py
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# Configure logging to file (no console output)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debate_log.txt')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Argument:
    """Represents a single argument in the debate"""
    agent: str
    round_num: int
    content: str
    timestamp: str

class DebateState(TypedDict):
    """State structure for the debate workflow"""
    topic: str
    current_round: int
    current_agent: str
    arguments: List[Dict[str, Any]]
    memory_summary: str
    debate_complete: bool
    winner: Optional[str]
    judgment_reason: Optional[str]
    full_summary: Optional[str]

class DebateSystem:
    def __init__(self, api_key: str):
        """Initialize the debate system with Gemini API"""
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0.7
        )
        self.graph = None
        self.build_graph()
    
    def user_input_node(self, state: DebateState) -> DebateState:
        """Node to handle user input for debate topic"""
        logger.info("=== USER INPUT NODE ===")
        
        if not state.get("topic"):
            topic = input("Enter topic for debate: ").strip()
            state["topic"] = topic
            state["current_round"] = 1
            state["current_agent"] = "Scientist"
            state["arguments"] = []
            state["memory_summary"] = f"Debate Topic: {topic}"
            state["debate_complete"] = False
            
            print(f"\n{'='*60}")
            print(f"🎭 DEBATE: Scientist vs Philosopher")
            print(f"📋 Topic: {topic}")
            print(f"{'='*60}\n")
        
        return state
    
    def scientist_agent_node(self, state: DebateState) -> DebateState:
        """Scientist agent node - evidence-based arguments"""
        logger.info(f"=== SCIENTIST AGENT NODE - Round {state['current_round']} ===")
        if state["current_agent"] != "Scientist":
            return state
        
        # Prepare context
        previous_args = "\n".join([
            f"[Round {arg['round_num']}] {arg['agent']}: {arg['content']}" 
            for arg in state["arguments"]
        ])
        
        prompt = f"""You are a Scientist participating in a structured debate.
        
Topic: {state['topic']}

Current Memory Summary: {state['memory_summary']}

Previous Arguments:
{previous_args}

This is Round {state['current_round']} of 8. You are making your {(state['current_round'] + 1) // 2} argument.

As a Scientist, base your argument on:
- Empirical evidence and data
- Scientific methodology
- Risk assessment and safety protocols
- Peer-reviewed research
- Quantifiable impacts

Make a compelling, evidence-based argument (2-3 sentences). Be persuasive but factual."""

        try:
            response = self.llm.invoke(prompt)
            argument_content = response.content.strip()
            
            # Create argument record
            argument = Argument(
                agent="Scientist",
                round_num=state["current_round"],
                content=argument_content,
                timestamp=datetime.now().isoformat()
            )
            
            # Update state
            state["arguments"].append(asdict(argument))

            # Display argument
            print(f"🔬[Round {state['current_round']}] Scientist:")
            print(f"    {argument_content}\n")
            
            # Update for next turn
            state["current_round"] += 1
            state["current_agent"] = "Philosopher"
            
        except Exception as e:
            logger.error(f"Error in scientist agent: {e}")
            print(f"Error in scientist agent: {e}")
            raise
        
        return state
    
    def philosopher_agent_node(self, state: DebateState) -> DebateState:
        """Philosopher agent node - conceptual and ethical arguments"""
        logger.info(f"=== PHILOSOPHER AGENT NODE - Round {state['current_round']} ===")
        
        if state["current_agent"] != "Philosopher":
            return state
        
        # Prepare context
        previous_args = "\n".join([
            f"[Round {arg['round_num']}] {arg['agent']}: {arg['content']}" 
            for arg in state["arguments"]
        ])
        
        prompt = f"""You are a Philosopher participating in a structured debate.
        
Topic: {state['topic']}

Current Memory Summary: {state['memory_summary']}

Previous Arguments:
{previous_args}

This is Round {state['current_round']} of 8. You are making your {state['current_round'] // 2} argument.

As a Philosopher, base your argument on:
- Ethical considerations and moral frameworks
- Historical precedents and lessons
- Conceptual analysis and definitions
- Social and cultural implications
- Individual rights and freedoms
- Long-term societal impact

Make a compelling, philosophically grounded argument (2-3 sentences). Be persuasive and thoughtful."""

        try:
            response = self.llm.invoke(prompt)
            argument_content = response.content.strip()
            
            # Create argument record
            argument = Argument(
                agent="Philosopher",
                round_num=state["current_round"],
                content=argument_content,
                timestamp=datetime.now().isoformat()
            )
            
            # Update state
            state["arguments"].append(asdict(argument))
            
            # Log and display
            logger.info(f"Philosopher argument: {argument_content}")
            logger.info(f"Philosopher argument: {argument_content}")
            print(f"🤔 [Round {state['current_round']}] Philosopher:")
            print(f"   {argument_content}\n")
            
            # Update for next turn
            if state["current_round"] < 8:
                state["current_round"] += 1
                state["current_agent"] = "Scientist"
            else:
                state["debate_complete"] = True
                state["current_agent"] = "Judge"
            
        except Exception as e:
            logger.error(f"Error in philosopher agent: {e}")
            print(f"Error in philosopher agent: {e}")
            raise
        
        return state
    
    def memory_node(self, state: DebateState) -> DebateState:
        """Node to update and maintain debate memory"""
        logger.info("=== MEMORY NODE ===")
        
        if not state["arguments"]:
            return state
        
        # Create structured summary of recent arguments
        recent_args = state["arguments"][-2:] if len(state["arguments"]) >= 2 else state["arguments"]
        
        summary_prompt = f"""Update the debate memory summary with the latest arguments.

Current Topic: {state['topic']}

Previous Summary: {state['memory_summary']}

Recent Arguments:
{chr(10).join([f"[Round {arg['round_num']}] {arg['agent']}: {arg['content']}" for arg in recent_args])}

Provide an updated summary that captures:
1. The main debate topic
2. Key points from both sides
3. Current trajectory of the debate
4. Notable patterns or themes

Keep it concise (3-4 sentences)."""

        try:
            response = self.llm.invoke(summary_prompt)
            updated_summary = response.content.strip()
            state["memory_summary"] = updated_summary
            
            logger.info(f"Memory updated: {updated_summary}")
            
        except Exception as e:
            logger.error(f"Error updating memory: {e}")
            # Keep previous summary if update fails
        
        return state
    
    def judge_node(self, state: DebateState) -> DebateState:
        """Judge node to evaluate the debate and declare winner"""
        logger.info("=== JUDGE NODE ===")
        
        if not state["debate_complete"]:
            return state
        
        # Prepare full debate transcript
        full_transcript = "\n".join([
            f"[Round {arg['round_num']}] {arg['agent']}: {arg['content']}" 
            for arg in state["arguments"]
        ])
        
        judgment_prompt = f"""You are an impartial judge evaluating a structured debate.

Topic: {state['topic']}

Full Debate Transcript:
{full_transcript}

Memory Summary: {state['memory_summary']}

As the judge, evaluate the debate based on:
1. Logical coherence and consistency
2. Quality and relevance of evidence
3. Persuasiveness of arguments
4. Addressing counterpoints
5. Overall strength of position

Provide:
1. A comprehensive summary of the debate (3-4 sentences)
2. The winner (either "Scientist" or "Philosopher")
3. Detailed reasoning for your decision (2-3 sentences)

Format your response as:
SUMMARY: [your summary]
WINNER: [Scientist or Philosopher]
REASON: [your reasoning]"""

        try:
            response = self.llm.invoke(judgment_prompt)
            judgment = response.content.strip()
            
            # Parse judgment
            lines = judgment.split('\n')
            summary_line = next((line for line in lines if line.startswith('SUMMARY:')), '')
            winner_line = next((line for line in lines if line.startswith('WINNER:')), '')
            reason_line = next((line for line in lines if line.startswith('REASON:')), '')
            
            state["full_summary"] = summary_line.replace('SUMMARY:', '').strip()
            state["winner"] = winner_line.replace('WINNER:', '').strip()
            state["judgment_reason"] = reason_line.replace('REASON:', '').strip()
            
            # Log judgment
            logger.info(f"Judgment complete - Winner: {state['winner']}")
            logger.info(f"Summary: {state['full_summary']}")
            logger.info(f"Reason: {state['judgment_reason']}")

            # Display results
            print(f"{'='*60}")
            print(f"\n[Judge] Summary of debate:")
            print(f"{'='*60}")
            print(f"{state['full_summary']}")
            print(f"\n[Judge] Winner: {state['winner']}")
            print(f"Reason: {state['judgment_reason']}")
            print(f"{'='*60}")
            
        except Exception as e:
            logger.error(f"Error in judge node: {e}")
            print(f"Error in judge node: {e}")
            state["winner"] = "Error in judgment"
            state["judgment_reason"] = "Could not complete evaluation"
        
        return state
    
    def should_continue_debate(self, state: DebateState) -> str:
        """Router function to determine next node"""
        if not state.get("topic"):
            return "user_input"
        elif state["debate_complete"]:
            return "judge"
        elif state["current_round"] <= 8:
            # Update memory after each argument
            if state["arguments"]:
                state = self.memory_node(state)
            
            if state["current_agent"] == "Scientist":
                return "scientist"
            else:
                return "philosopher"
        else:
            return END
    
    def build_graph(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(DebateState)
        
        # Add nodes
        workflow.add_node("user_input", self.user_input_node)
        workflow.add_node("scientist", self.scientist_agent_node)
        workflow.add_node("philosopher", self.philosopher_agent_node)
        workflow.add_node("judge", self.judge_node)
        
        # Add edges
        workflow.set_entry_point("user_input")
        workflow.add_conditional_edges(
            "user_input",
            self.should_continue_debate,
            {
                "user_input": "user_input",
                "scientist": "scientist",
                "philosopher": "philosopher",
                "judge": "judge"
            }
        )
        
        workflow.add_conditional_edges(
            "scientist",
            self.should_continue_debate,
            {
                "scientist": "scientist",
                "philosopher": "philosopher",
                "judge": "judge"
            }
        )
        
        workflow.add_conditional_edges(
            "philosopher",
            self.should_continue_debate,
            {
                "scientist": "scientist",
                "philosopher": "philosopher",
                "judge": "judge"
            }
        )
        
        workflow.add_edge("judge", END)
        
        self.graph = workflow.compile()


    def show_workflow_diagram(self):
        """Display the workflow DAG diagram and save as image"""
        try:
            # Save the diagram as PNG file
            diagram_data = self.graph.get_graph().draw_mermaid_png()
            
            # Save to file
            with open('debate_workflow_diagram.png', 'wb') as f:
                f.write(diagram_data)
            
            print("🔄 Workflow DAG Diagram saved as 'debate_workflow_diagram.png'")
            print("=" * 50)
        except Exception as e:
            print(f"Could not generate diagram: {e}")
            print("📊 Workflow: user_input → scientist ⇄ philosopher → judge → END")

    def run_debate(self) -> DebateState:
        """Execute the debate workflow"""
        logger.info("Starting debate system...")

        # Show workflow diagram
        self.show_workflow_diagram()
        print("\n")
        
        initial_state = DebateState(
            topic="",
            current_round=1,
            current_agent="Scientist",
            arguments=[],
            memory_summary="",
            debate_complete=False,
            winner=None,
            judgment_reason=None,
            full_summary=None
        )
        
        try:
            final_state = self.graph.invoke(initial_state)
            
            # Save final log
            self.save_debate_log(final_state)
            
            return final_state
            
        except Exception as e:
            logger.error(f"Error running debate: {e}")
            print(f"Error running debate: {e}")
            raise
    
    def save_debate_log(self, state: DebateState):
        """Save complete debate log to file"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "topic": state["topic"],
            "arguments": state["arguments"],
            "memory_summary": state["memory_summary"],
            "full_summary": state["full_summary"],
            "winner": state["winner"],
            "judgment_reason": state["judgment_reason"]
        }
        
        with open('complete_debate_log.json') as f:
            json.dump(log_data, f, indent=2)
        
        logger.info("Complete debate log saved to complete_debate_log.json")