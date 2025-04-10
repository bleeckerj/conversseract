# autogen_service.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import json
import uuid
import os
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager, Cache
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen_core.model_context import BufferedChatCompletionContext, TokenLimitedChatCompletionContext
import asyncio
import logging
from contextlib import asynccontextmanager
import requests
from datetime import datetime
import pathlib
import random

# Configure logging with both console and file output
import logging
import os
from datetime import datetime
import random
import uuid

# Create logs directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Create a unique log filename with timestamp
log_filename = os.path.join(log_dir, f"conversseract_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Configure root logger
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[
                       logging.FileHandler(log_filename),
                       logging.StreamHandler()  # This should go to console
                   ])

logger = logging.getLogger(__name__)

# Print log location in a way that will definitely be visible
print("\n" + "="*80)
print(f"LOGS ARE BEING WRITTEN TO: {os.path.abspath(log_filename)}")
print("="*80 + "\n")

# Store active debates
active_debates = {}

# Global dictionary for storing debate results
debate_results = {}

# Add this as a global variable near the top
default_moderator_system_message = "You are moderating a debate. Summarize positions, highlight agreements and disagreements, and pose questions to deepen the discussion."

# Load LLM configuration
def load_llm_config(api_key=None, model_provider="openai", model_name="gpt-4", base_url=None):
    """
    Load LLM configuration based on selected provider and model.
    
    Args:
        api_key: API key for hosted models
        model_provider: "openai", "anthropic", "ollama", etc.
        model_name: Specific model to use
        base_url: Base URL for API requests (needed for Ollama, Azure, etc.)
    """
    # Set API key for hosted models if provided
    if api_key and model_provider in ["openai", "anthropic"]:
        os.environ[f"{model_provider.upper()}_API_KEY"] = api_key
    
    # Configure for different providers
    if model_provider == "ollama":
        # Simplified Ollama configuration with smaller token limits
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        random_suffix = str(uuid.uuid4())[:8]
        
        # Use directly supported parameters only
        return {
            "config_list": [{
                "model": model_name,
                "base_url": base_url or "http://localhost:11434/v1",
                "api_type": "ollama",
            }],
            "temperature": 0.7 + (random.random() * 0.1),
            "max_tokens": 500  # Reduced from 1000 to stay within context limits
        }
    elif model_provider == "openai":
        # OpenAI configuration
        return {
            "config_list": [{
                "model": model_name,
                "api_key": os.environ.get("OPENAI_API_KEY"),
                "api_type": "openai",
            }],
            "temperature": 0.7,
            "max_tokens": 1000
        }
    else:
        # Generic configuration for other providers
        # This will need customization based on specific providers
        return {
            "config_list": [{
                "model": model_name,
                "api_key": api_key,
                "base_url": base_url,
            }],
            "temperature": 0.7,
            "max_tokens": 1000
        }

# Add this helper function to handle Ollama-specific logic

def customize_for_ollama(agent, llm_config):
    """Apply Ollama-specific customizations to an agent"""
    # Check if using Ollama
    if "config_list" in llm_config and llm_config["config_list"]:
        config = llm_config["config_list"][0]
        if "api_base" in config and "localhost:11434" in config.get("api_base", ""):
            # Customize Ollama behavior
            
            # 1. Shorter outputs for local models to reduce hang risk
            llm_config["max_tokens"] = min(llm_config.get("max_tokens", 1000), 750)
            
            # 2. Adjust temperature based on model capabilities
            llm_config["temperature"] = 0.8
            
            # 3. For retrieval agents, use smaller chunk sizes
            if hasattr(agent, "retrieve_config"):
                agent.retrieve_config["chunk_token_size"] = 500
            
            logger.info(f"Customized agent {agent.name} for Ollama use")
    
    return agent

# Add after the customize_for_ollama function

def create_message_transformer(strategy="sliding_window", window_size=10, 
                              compression_level="medium", topic=""):
    """Create a message transformation function for context management
    
    Args:
        strategy: "sliding_window", "compress_history", "summarize_all", or "hybrid"
        window_size: Number of recent messages to keep in full
        compression_level: "low", "medium", "high" - how aggressively to compress
        topic: The debate topic for context preservation
    """
    
    def sliding_window_filter(messages):
        """Keep the first message (topic) and the most recent n messages"""
        if len(messages) <= window_size:
            return messages
        # Always keep the first message (topic) + most recent messages
        return [messages[0]] + messages[-(window_size-1):]
    
    def compress_history(messages):
        """Compress older messages while keeping recent ones intact"""
        if len(messages) <= window_size:
            return messages
            
        # Keep recent messages as they are
        recent_messages = messages[-window_size:]
        
        # Compress older messages
        older_messages = messages[1:-window_size]  # Skip the first message (topic)
        
        # Group older messages by speaker for better summarization
        compressed = []
        current_speaker = None
        current_content = []
        
        for msg in older_messages:
            speaker = msg.get("sender", "Unknown")
            content = msg.get("content", "")
            
            if speaker != current_speaker and current_content:
                # Add the compressed message for previous speaker
                compressed.append({
                    "sender": current_speaker,
                    "content": f"[SUMMARY] {current_speaker} made these points: " + 
                              "; ".join(current_content)
                })
                current_content = []
            
            current_speaker = speaker
            current_content.append(content.split("\n")[0])  # Take first line as summary
        
        # Add the last speaker if any
        if current_speaker and current_content:
            compressed.append({
                "sender": current_speaker,
                "content": f"[SUMMARY] {current_speaker} made these points: " + 
                          "; ".join(current_content)
            })
            
        # Return the topic + compressed history + recent messages
        return [messages[0]] + compressed + recent_messages
    
    # Return the selected strategy
    if strategy == "sliding_window":
        return sliding_window_filter
    elif strategy == "compress_history":
        return compress_history
    else:
        # Default to sliding window
        return sliding_window_filter

# Models
class Document(BaseModel):
    content: str
    metadata: Dict[str, str] = {}

class KnowledgeBase(BaseModel):
    id: str
    name: str
    documents: List[Document] = []

class Agent(BaseModel):
    id: str
    name: str
    personality: str
    expertise: List[str]
    position: Optional[str] = None
    system_message: Optional[str] = None
    model: str = "gpt-4o-mini"  # Default model name
    model_provider: Optional[str] = None  # If None, use the debate's provider
    knowledge_base_id: Optional[str] = None
    enable_reflection: bool = False
    reflection_depth: int = 1
    # New fields for model configuration
    model_base_url: Optional[str] = None  # Override base URL for this agent
    model_api_key: Optional[str] = None   # Override API key for this agent
    max_tokens: Optional[int] = None  # New field

class DebateConfig(BaseModel):
    topic: str
    agent_ids: List[str]
    rounds: int = 3
    moderator_enabled: bool = True
    initial_prompt: Optional[str] = None
    api_key: Optional[str] = None
    speaker_selection_method: str = "round_robin"  # Options: "round_robin", "auto", "random"
    model_provider: str = "openai"  # Options: "openai", "anthropic", "ollama", etc.
    model_base_url: Optional[str] = None  # For Ollama: "http://localhost:11434/v1"
    default_model: str = "gpt-4"  # Model to use if not specified by agent
    
    # New memory management options
    context_strategy: str = "sliding_window"  # "sliding_window", "compress_history", "hybrid"
    context_window_size: int = 10  # How many messages to keep in full
    infinite_debate: bool = False  # If true, override rounds limit

class Message(BaseModel):
    """A single message in a debate."""
    agent: Dict[str, Any]  # Changed from str to Dict[str, Any
    content: str
    referenced_docs: List[str] = []
    observation: Optional[str] = None

class DebateResponse(BaseModel):
    debate_id: str
    status: str
    messages: List[Message] = []
    is_complete: bool = False

# Add this after your model definitions

class ReflectiveAssistantAgent(AssistantAgent):
    """An agent that internally reflects on the conversation before responding"""
    
    def __init__(self, name, system_message, reflection_depth=1, **kwargs):
        super().__init__(name=name, system_message=system_message, **kwargs)
        self.reflection_depth = reflection_depth
        self.observations = []
        
    async def _generate_response(self, messages, sender):
        """Override the response generation to include a reflection step"""
        # First, reflect on the conversation
        reflection = await self._reflect(messages)
        self.observations.append(reflection)
        
        # Enhance the system message with reflection
        enhanced_messages = messages.copy()
        
        # Add reflection as a private thought
        reflection_msg = {
            "role": "system",
            "content": f"Before responding, reflect on the conversation: {reflection}\n\nIncorporate this reflection into your response, but do NOT explicitly mention that you've done any internal reflection."
        }
        
        # Add reflection right before the last user message
        for i in range(len(enhanced_messages) - 1, -1, -1):
            if enhanced_messages[i]["role"] != "assistant":
                enhanced_messages.insert(i + 1, reflection_msg)
                break
        
        # Generate response with the enhanced context
        return await super()._generate_response(enhanced_messages, sender)
    
    async def _reflect(self, messages):
        """Generate an internal reflection based on the conversation history"""
        # Create a prompt for reflection
        reflection_prompt = {
            "role": "system",
            "content": """Please reflect on the conversation so far. Consider:
1. What are the key points made by each participant?
2. How can I contribute most constructively?
3. What knowledge would strengthen my response?

Keep your reflection concise."""
        }
        
        # Simplify for local models if needed
        if self.llm_config.get("config_list", [{}])[0].get("api_base", "").find("localhost") >= 0:
            reflection_prompt["content"] = "Summarize the key points in the conversation and identify how you can contribute."
        
        # Add the conversation history
        history = [
            msg for msg in messages 
            if msg["role"] in ["user", "assistant"]
        ]
        
        # Combine prompts
        reflection_messages = [reflection_prompt] + history
        
        # Generate the reflection using the LLM
        llm_config = self.llm_config.copy()
        # Optionally use a cheaper model for reflection if needed
        
        client = self._create_client(llm_config)
        response = await client.create(
            messages=reflection_messages,
            temperature=0.6,  # Lower temperature for more analytical reflection
            max_tokens=500
        )
        
        return response.choices[0].message.content

class ReflectiveRetrieveAgent(RetrieveUserProxyAgent):
    """A retrieval agent that reflects on the conversation before responding"""
    
    def __init__(self, name, system_message, reflection_depth=1, **kwargs):
        super().__init__(name=name, system_message=system_message, **kwargs)
        self.reflection_depth = reflection_depth
        self.observations = []
    
    async def _process_received_message(self, message, sender, silent):
        """Add reflection before processing the message"""
        # Generate reflection first
        messages = self._oai_messages[sender]
        reflection = await self._reflect(messages)
        self.observations.append(reflection)
        
        # Add the reflection as guidance for retrieval and response
        reflection_msg = {
            "role": "system", 
            "content": f"Before responding, I've reflected on the conversation: {reflection}\n\nUse this reflection to guide your retrieval and response. Incorporate insights from this reflection, but do NOT explicitly mention that you've reflected."
        }
        
        # Insert the reflection as a system message
        self._oai_messages[sender].append(reflection_msg)
        
        # Continue with normal processing
        return await super()._process_received_message(message, sender, silent)
    
    async def _reflect(self, messages):
        """Generate an internal reflection based on the conversation history"""
        # Create a prompt for reflection
        reflection_prompt = {
            "role": "system",
            "content": """Please reflect on the conversation so far. Consider:
1. What are the key points made by each participant?
2. How can I contribute most constructively?
3. What knowledge would strengthen my response?

Keep your reflection concise."""
        }
        
        # Simplify for local models if needed
        if self.llm_config.get("config_list", [{}])[0].get("api_base", "").find("localhost") >= 0:
            reflection_prompt["content"] = "Summarize the key points in the conversation and identify how you can contribute."
        
        # Add the conversation history
        history = [
            msg for msg in messages 
            if msg["role"] in ["user", "assistant", "system"] and 
            "Before responding, I've reflected" not in msg.get("content", "")
        ]
        
        # Combine prompts
        reflection_messages = [reflection_prompt] + history
        
        # Generate the reflection using the LLM
        llm_config = self.llm_config.copy()
        
        client = self._create_client(llm_config)
        response = await client.create(
            messages=reflection_messages,
            temperature=0.3,
            max_tokens=500
        )
        
        return response.choices[0].message.content

# Background tasks for async debate processing
async def run_debate(debate_id: str, config: DebateConfig, knowledge_bases: Dict[str, KnowledgeBase], 
                    agents: Dict[str, Agent]):
    logger.info("=" * 50)
    logger.info(f"DEBATE DEBUG: Starting debate {debate_id} with {config.rounds} rounds")
    logger.info("=" * 50)
    logger.info(f"DEBUG: Starting NEW debate with ID {debate_id}")
    logger.info(f"Starting debate {debate_id} on topic: {config.topic}")
    logger.info(f"Starting debate with max_round={config.rounds} and selection method={config.speaker_selection_method}")
    
    # Get the default model config
    default_llm_config = load_llm_config(
        api_key=config.api_key,
        model_provider=config.model_provider,
        model_name=config.default_model,
        base_url=config.model_base_url
    )
    
    # Create AutoGen agents
    autogen_agents = []
    logger.info(f"DEBUG: Requested agent IDs: {config.agent_ids}")
    logger.info(f"DEBUG: Available agent IDs: {list(agents.keys())}")
    agent_map = {}
    
    # Create the moderator only if enabled
    if config.moderator_enabled:
        moderator = UserProxyAgent(
            name="Moderator",
            human_input_mode="NEVER",
            system_message=default_moderator_system_message,
            llm_config=default_llm_config,
            code_execution_config=False
        )
        autogen_agents.append(moderator)
        agent_map["Moderator"] = moderator
    
    # Create debate participants
    for agent_id in config.agent_ids:
        if agent_id not in agents:
            logger.error(f"Agent {agent_id} not found")
            continue
            
        agent_config = agents[agent_id]
        
        # Use agent-specific model if defined, otherwise use debate default
        model_provider = agent_config.model_provider or config.model_provider
        model_name = agent_config.model
        model_base_url = agent_config.model_base_url or config.model_base_url
        model_api_key = agent_config.model_api_key or config.api_key
        
        # Load agent-specific LLM config
        llm_config = load_llm_config(
            api_key=model_api_key,
            model_provider=model_provider,
            model_name=model_name,
            base_url=model_base_url
        )
        
        # Modify llm_config for each agent
        if agent_config.max_tokens:
            llm_config["max_tokens"] = agent_config.max_tokens
        
        # Build system message
        system_message = agent_config.system_message or f"""
        You are {agent_config.name}, an AI assistant with expertise in {', '.join(agent_config.expertise)}.
        Your personality can be described as: {agent_config.personality}.
        Your response should be conversational rather than formal, as if you are extemporaneously chatting with others.
        """
        
        if agent_config.position:
            system_message += f"\nIn this salon discussion, you are arguing for the position: {agent_config.position}."
            
        # Example 1: Limit messages
        buffered_context = BufferedChatCompletionContext(buffer_size=5)

        # Example 2: Limit tokens
        ### token_limited_context = TokenLimitedChatCompletionContext(token_limit=1000)

        # If there's a knowledge base, set up a retrieval agent
        if agent_config.knowledge_base_id and agent_config.knowledge_base_id in knowledge_bases:
            kb = knowledge_bases[agent_config.knowledge_base_id]
            
            # Format documents for RAG
            documents = [doc.content for doc in kb.documents]
            
            # Create a reflective retrieval-enabled agent if reflection is enabled
            if agent_config.enable_reflection:
                agent = ReflectiveRetrieveAgent(
                    name=agent_config.name,
                    system_message=system_message,
                    human_input_mode="NEVER",
                    llm_config=llm_config,
                    reflection_depth=agent_config.reflection_depth,
                    retrieve_config={
                        "docs": documents,
                        "chunk_token_size": 1000,
                        "model": agent_config.model,
                    }
                )
            else:
                # Standard retrieval agent without reflection
                agent = RetrieveUserProxyAgent(
                    name=agent_config.name,
                    system_message=system_message,
                    human_input_mode="NEVER",
                    llm_config=llm_config,
                    retrieve_config={
                        "docs": documents,
                        "chunk_token_size": 1000,
                        "model": agent_config.model,
                    }
                )
        else:
            # Create either a reflective or standard assistant agent
            if agent_config.enable_reflection:
                agent = ReflectiveAssistantAgent(
                    name=agent_config.name,
                    system_message=system_message,
                    llm_config=llm_config,
                    reflection_depth=agent_config.reflection_depth
                )
            else:
                print(f"Using LLM config: {llm_config}")
                
                agent = AssistantAgent(
                    name=agent_config.name,
                    system_message=system_message,
                    llm_config=llm_config,
                )
        
        # Apply Ollama-specific customizations if needed
        agent = customize_for_ollama(agent, llm_config)
                
        autogen_agents.append(agent)
        agent_map[agent_config.name] = agent
    
    # Create the group chat with the specified selection method
    # Set up infinite rounds if requested
    max_rounds = 1000000 if config.infinite_debate else config.rounds

    # Set up message transformation
    message_transformer = create_message_transformer(
        strategy=config.context_strategy,
        window_size=config.context_window_size,
        topic=config.topic
    )

    # Create the group chat with ONLY parameters that actually exist
    group_chat = GroupChat(
        agents=autogen_agents,
        messages=[],
        max_round=max_rounds,
        speaker_selection_method=config.speaker_selection_method
    )
    
    # Add round tracking
    def create_debug_wrapper(original_function, max_rounds):
        """Create a debug wrapper for speaker selection method"""
        def wrapper(*args, **kwargs):
            # The first arg should be the GroupChat object
            if args and hasattr(args[0], '_round_index'):
                group_chat = args[0]
                round_index = getattr(group_chat, '_round_index', 0)
                logger.info(f"DEBUG: Round {round_index}/{max_rounds}: Selecting next speaker")
                
                # Call original method
                speaker = original_function(*args, **kwargs)
                
                # Log result
                logger.info(f"DEBUG: Selected speaker: {speaker}, is_terminal_round: {round_index + 1 >= max_rounds}")
                return speaker
            else:
                # Fallback if something's wrong with the arguments
                logger.warning("DEBUG: GroupChat object not found in arguments")
                return original_function(*args, **kwargs)
        
        return wrapper

    # Apply the debug wrapper to group_chat's speaker selection
    original_function = group_chat.a_select_speaker
    group_chat.a_select_speaker = create_debug_wrapper(original_function, config.rounds)

    # Create the manager with the wrapped group_chat
    manager = GroupChatManager(groupchat=group_chat)

    # Add additional explicit debugging logs
    logger.info(f"DEBUG: Created GroupChat with max_round={config.rounds} and selection_method={config.speaker_selection_method}")
    
    # Make sure we select the correct initiator (not always the first agent)
    if config.moderator_enabled and "Moderator" in agent_map:
        initiator = agent_map["Moderator"]
    elif autogen_agents:
        initiator = autogen_agents[0]  # First participant if no moderator
    else:
        initiator = None
    
    # Start the debate
    initial_message = config.initial_prompt or f"Let's have a salon-like discussion around this topic: '{config.topic}'"

    # Log all available agents
    for i, agent in enumerate(autogen_agents):
        logger.info(f"DEBUG: Agent {i+1}: {agent.name}, type: {agent.__class__.__name__}")

    # Log the configured max rounds vs number of agents
    logger.info(f"DEBUG: Using {config.speaker_selection_method} selection with {config.rounds} rounds for {len(autogen_agents)} agents")
    if config.speaker_selection_method == "round_robin" and config.rounds < len(autogen_agents):
        logger.warning(f"WARN: Not enough rounds ({config.rounds}) for all {len(autogen_agents)} agents to speak with round_robin selection!")

    try:
        logger.info(f"DEBATE DEBUG: About to start chat with {len(autogen_agents)} agents")
        logger.info(f"DEBATE DEBUG: Initial message: {initial_message}")
        logger.info(f"DEBATE DEBUG: GroupChat has max_round={group_chat.max_round}")
        if initiator:
            with Cache.disk(cache_seed=None) as cache:  # Disable caching
                await asyncio.to_thread(
                    initiator.initiate_chat, 
                    manager,
                    message=initial_message,
                    cache=cache
                )
            
            logger.info(f"Debate completed with {len(manager.groupchat.messages)} total messages")
            
            # Collect all messages
            all_messages = []
            for msg in manager.groupchat.messages:
                sender_name = msg.get("name", "Unknown")
                content = msg.get("content", "")
                
                # Get more detailed agent info if available
                agent_info = {
                    "name": sender_name,
                    "agent_type": "unknown"
                }
                
                # Try to match with an actual agent object
                if sender_name in agent_map:
                    agent = agent_map[sender_name]
                    agent_info = {
                        "name": agent.name,
                        "agent_type": agent.__class__.__name__,
                        "system_message_summary": agent.system_message[:100] + "..." if hasattr(agent, "system_message") else None
                    }
                else:
                    logger.warning(f"Could not match sender '{sender_name}' to any agent in agent_map")
                
                # Extract any referenced documents or observations
                referenced_docs = []
                observation = None
                
                all_messages.append(Message(
                    agent=agent_info,
                    content=content,
                    referenced_docs=referenced_docs,
                    observation=observation
                ))
            
            # After a debate completes, all messages are collected and stored here:
            debate_results[debate_id] = {
                "status": "completed",
                "messages": all_messages,
                "is_complete": True
            }
            
            # Save debate results to file
            save_debate_to_file(debate_id, debate_results[debate_id])
        else:
            logger.error(f"No agents available for debate {debate_id}")
            debate_results[debate_id] = {
                "status": "error",
                "messages": [],
                "is_complete": True
            }
    except Exception as e:
        logger.exception(f"Error in debate {debate_id}: {str(e)}")
        debate_results[debate_id] = {
            "status": "error",
            "messages": [Message(agent="System", content=f"Error: {str(e)}")],
            "is_complete": True
        }
    
    # Remove from active debates
    if debate_id in active_debates:
        del active_debates[debate_id]

# Add this helper function to your code

def fix_newlines_in_messages(messages):
    """Fix escaped newlines in message content to improve readability"""
    for message in messages:
        if "content" in message and isinstance(message["content"], str):
            # Replace escaped newlines with actual newlines
            message["content"] = message["content"].replace("\\n\\n", "\n\n")
            # Also handle single newlines
            message["content"] = message["content"].replace("\\n", "\n")
    return messages

# Then use this function when saving debate results to file
def save_debate_to_file(debate_id: str, result: dict):
    """Save debate results to a JSON file"""
    # Create a debates directory if it doesn't exist
    debates_dir = pathlib.Path("debates")
    debates_dir.mkdir(exist_ok=True)
    
    # Format filename with date and debate ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = debates_dir / f"debate_{timestamp}_{debate_id}.json"
    
    # Fix newlines in message content
    if "messages" in result:
        result["messages"] = fix_newlines_in_messages(result["messages"])
    
    # Convert message objects to dictionaries and save
    debate_data = {
        "debate_id": debate_id,
        "status": result.get("status", "unknown"),
        "is_complete": result.get("is_complete", False),
        "messages": [msg.dict() if hasattr(msg, "dict") else msg for msg in result.get("messages", [])],
        "timestamp": timestamp
    }
    
    # Write to file with proper JSON formatting to preserve newlines
    with open(filename, "w") as f:
        json.dump(debate_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved debate {debate_id} to {filename}")
    return str(filename)

# Initialize the FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load any persistent data
    yield
    # Shutdown: clean up resources
    for debate_id in list(active_debates.keys()):
        logger.info(f"Cleaning up debate {debate_id}")
        # Ideally we'd cancel any running task here

app = FastAPI(lifespan=lifespan)

# Create storage for knowledge bases and agents
knowledge_bases = {}
agents = {}

# API endpoints
@app.post("/knowledge-bases", response_model=KnowledgeBase)
async def create_knowledge_base(kb: KnowledgeBase):
    knowledge_bases[kb.id] = kb
    return kb

@app.post("/knowledge-bases/{kb_id}/documents")
async def add_document(kb_id: str, document: Document):
    if kb_id not in knowledge_bases:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    knowledge_bases[kb_id].documents.append(document)
    return {"status": "success"}

@app.post("/agents", response_model=Agent)
async def create_agent(agent: Agent):
    agents[agent.id] = agent
    logger.debug(f"Created agent {agent} with ID {agent.id}")
    return agent

@app.put("/agents/{agent_id}/system-message")
async def update_agent_system_message(agent_id: str, system_message: str):
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agents[agent_id].system_message = system_message
    return {"status": "success", "agent_id": agent_id}

@app.put("/agents/{agent_id}/toggle-reflection")
async def toggle_agent_reflection(agent_id: str, enable: bool, depth: int = 1):
    """Enable or disable agent reflection capabilities"""
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agents[agent_id].enable_reflection = enable
    agents[agent_id].reflection_depth = depth
    
    return {
        "status": "success", 
        "agent_id": agent_id,
        "reflection_enabled": enable,
        "reflection_depth": depth
    }

@app.put("/debates/moderator-system-message")
async def update_moderator_system_message(system_message: str):
    global default_moderator_system_message
    default_moderator_system_message = system_message
    return {"status": "success"}

@app.get("/debate-settings")
async def get_debate_settings():
    """Return available speaker selection methods and other settings"""
    return {
        "speaker_selection_methods": ["round_robin", "auto", "random"],
        "default_rounds": 3,
        "moderator_system_message": default_moderator_system_message,
        "model_providers": [
            {
                "id": "openai",
                "name": "OpenAI",
                "default_models": ["gpt-4", "gpt-3.5-turbo"]
            },
            {
                "id": "ollama",
                "name": "Ollama (Local)",
                "default_models": ["llama3", "llama2", "mistral", "mixtral"]
            },
            {
                "id": "anthropic",
                "name": "Anthropic",
                "default_models": ["claude-3-opus", "claude-3-sonnet"]
            }
        ]
    }

@app.post("/debates", response_model=DebateResponse)
async def start_debate(config: DebateConfig, background_tasks: BackgroundTasks):
    # Generate a unique ID for this debate
    debate_id = str(uuid.uuid4())
    
    # Initialize debate status
    active_debates[debate_id] = "starting"
    debate_results[debate_id] = {
        "status": "in_progress",
        "messages": [],
        "is_complete": False
    }
    
    # Start the debate in the background
    background_tasks.add_task(run_debate, debate_id, config, knowledge_bases, agents)
    
    # Return initial response
    return DebateResponse(
        debate_id=debate_id,
        status="starting",
        messages=[],
        is_complete=False
    )

@app.get("/debates/{debate_id}", response_model=DebateResponse)
async def get_debate_status(debate_id: str):
    # Returns all debate messages
    if debate_id not in active_debates and debate_id not in debate_results:
        raise HTTPException(status_code=404, detail="Debate not found")
    
    result = debate_results.get(debate_id, {})
    return DebateResponse(
        debate_id=debate_id,
        status=result.get("status", "unknown"),
        messages=result.get("messages", []),
        is_complete=result.get("is_complete", False)
    )

@app.get("/debates/{debate_id}/observations")
async def get_debate_observations(debate_id: str):
    # Returns just the internal observations/reflections
    if debate_id not in debate_results:
        raise HTTPException(status_code=404, detail="Debate not found")
    
    result = debate_results.get(debate_id, {})
    messages = result.get("messages", [])
    observations = [
        {
            "agent": msg.agent,
            "message_content": msg.content,
            "observation": msg.observation
        }
        for msg in messages
        if msg.observation is not None
    ]
    return {"debate_id": debate_id, "observations": observations}

@app.get("/model-providers/ollama/models")
async def list_ollama_models(base_url: str = "http://localhost:11434"):
    """List available models from Ollama server"""
    try:
        # Call Ollama API to list models
        response = requests.get(f"{base_url}/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            return {
                "status": "success",
                "models": [model.get("name") for model in models]
            }
        else:
            return {
                "status": "error",
                "message": f"Ollama returned status code {response.status_code}"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error connecting to Ollama: {str(e)}"
        }

@app.get("/examples/ollama-client")
async def get_ollama_example():
    """Get example code for using Ollama with the service"""
    example_code = """
import requests
import json

# Base URL for the ConversSeract API
BASE_URL = "http://localhost:8000"

# 1. Create two agents
agents = [
    {
        "id": "tech_enthusiast",
        "name": "Alex",
        "personality": "enthusiastic, optimistic, tech-savvy",
        "expertise": ["artificial intelligence", "programming", "tech trends"],
        "model": "llama3",  # Ollama model name
        "model_provider": "ollama"
    },
    {
        "id": "philosopher",
        "name": "Jordan",
        "personality": "thoughtful, analytical, concerned with ethics",
        "expertise": ["philosophy", "ethics", "social impact"],
        "model": "llama3",  # Ollama model name
        "model_provider": "ollama"
    }
]

# Create the agents
for agent in agents:
    response = requests.post(f"{BASE_URL}/agents", json=agent)
    print(f"Created agent {agent['name']}: {response.status_code}")

# 2. Start a debate
debate_config = {
    "topic": "The impact of AI on society",
    "agent_ids": ["tech_enthusiast", "philosopher"],
    "rounds": 5,
    "model_provider": "ollama",
    "model_base_url": "http://localhost:11434/v1",
    "default_model": "llama3"
}

response = requests.post(f"{BASE_URL}/debates", json=debate_config)
debate_id = response.json()["debate_id"]
print(f"Started debate: {debate_id}")

# 3. Poll for results
import time
while True:
    response = requests.get(f"{BASE_URL}/debates/{debate_id}")
    result = response.json()
    
    if result["is_complete"]:
        print("Debate completed!")
        for msg in result["messages"]:
            print(f"{msg['agent']}: {msg['content'][:100]}...")
        break
        
    print(f"Debate in progress... {len(result['messages'])} messages so far")
    time.sleep(5)
"""
    return {"example_code": example_code}

# Add new endpoint for terminating a debate

@app.post("/debates/{debate_id}/terminate")
async def terminate_debate(debate_id: str):
    """Manually terminate an ongoing debate"""
    if debate_id not in active_debates:
        raise HTTPException(status_code=404, detail="No active debate with this ID")
    
    # Mark the debate as complete
    if debate_id in debate_results:
        debate_results[debate_id]["status"] = "terminated"
        debate_results[debate_id]["is_complete"] = True
    
    # Remove from active debates
    del active_debates[debate_id]
    
    return {"status": "success", "message": f"Debate {debate_id} terminated"}

# Update the example code

@app.get("/examples/infinite-debate")
async def get_infinite_debate_example():
    """Get example code for running an infinite debate"""
    example_code = """
import requests
import json

# Base URL for the ConversSeract API
BASE_URL = "http://localhost:8000"

# Configuration for infinite debate
debate_config = {
    "topic": "The future of artificial intelligence",
    "agent_ids": ["tech_enthusiast", "philosopher", "skeptic"],
    "rounds": 100,  # This will be ignored due to infinite_debate=True
    "infinite_debate": True,
    "context_strategy": "compress_history",
    "context_window_size": 8,
    "speaker_selection_method": "round_robin",
    "model_provider": "ollama",
    "model_base_url": "http://localhost:11434/v1",
    "default_model": "llama3"
}

# Start the debate
response = requests.post(f"{BASE_URL}/debates", json=debate_config)
debate_id = response.json()["debate_id"]
print(f"Started infinite debate: {debate_id}")

# Poll for messages and allow manual termination
import time
try:
    while True:
        response = requests.get(f"{BASE_URL}/debates/{debate_id}")
        result = response.json()
        
        if result["is_complete"]:
            print("Debate completed or terminated!")
            break
            
        print(f"Debate in progress... {len(result['messages'])} messages so far")
        user_input = input("Press Enter to continue polling, or 'stop' to terminate the debate: ")
        
        if user_input.lower() == 'stop':
            # Terminate the debate
            requests.post(f"{BASE_URL}/debates/{debate_id}/terminate")
            print("Termination request sent")
            break
            
        time.sleep(5)
except KeyboardInterrupt:
    # Handle Ctrl+C to terminate debate
    requests.post(f"{BASE_URL}/debates/{debate_id}/terminate")
    print("\\nDebate terminated by user")
"""
    return {"example_code": example_code}

# Main entry point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("autogen_conversseract_service:app", host="0.0.0.0", port=port, reload=True)

# test_ollama.py
import requests
import ollama

print("Testing Ollama API...")
try:
    # Test direct API
    response = requests.get("http://localhost:11434/api/tags")
    print(f"API response: {response.status_code}")
    print(f"Models: {[m['name'] for m in response.json().get('models', [])]}")
    
    # Test client library
    models = ollama.list()
    print(f"Client models: {models}")
    
    # Test simple completion
    response = ollama.chat(model="phi3:mini", messages=[
        {"role": "user", "content": "Hello! How are you today?"}
    ])
    print(f"Response: {response['message']['content'][:100]}...")
    
    print("Ollama is working correctly!")
except Exception as e:
    print(f"Error: {str(e)}")