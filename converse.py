import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import json

# Load LLM configuration
with open("config.json", "r") as f:
    llm_config = json.load(f)

# Create the dinner party guests with more casual personalities
tech_enthusiast = AssistantAgent(
    name="Alex",
    system_message="You're a follower of Ralph Waldo Emerson and a transcendentalist. Be conversational and friendly.",
    llm_config=llm_config
)

humanities_professor = AssistantAgent(
    name="Jordan",
    system_message="You're a German emigré who lives in St. Louis in the 1860s. Share thoughts casually, ask questions, and refer to books or ideas you've encountered.",
    llm_config=llm_config
)

entrepreneur = AssistantAgent(
    name="Taylor",
    system_message="You're an entrepreneur with experience in tech startups. You're practical and business-minded but also care about creating meaningful products. Share startup stories, business insights, and your vision for how technology can help people.",
    llm_config=llm_config
)

# Create a user proxy agent as the host
host = UserProxyAgent(
    name="Sam",
    human_input_mode="NEVER",  # This makes it automated without requiring human input
    system_message="You're hosting a dinner party. Keep the conversation flowing naturally, ask open-ended questions, and occasionally bring up new topics related to technology and society. Be warm and inclusive, making sure everyone gets a chance to speak.",
    llm_config=llm_config,
    code_execution_config=False
)

# Set up the group chat with more flexible structure
dinner_conversation = GroupChat(
    agents=[host, tech_enthusiast, humanities_professor, entrepreneur],
    messages=[],
    max_round=15,  # Allow for a longer, more meandering conversation
    speaker_selection_method="round_robin"  # More natural turn-taking
)

# Create the manager
manager = GroupChatManager(
    groupchat=dinner_conversation,
    llm_config=llm_config
)

# Start the conversation with a more casual opener
host.initiate_chat(
    manager,
    message="Thanks for coming to dinner tonight! I thought we could chat about how AI and technology are changing our everyday lives. Alex was just telling me about a new AI tool they've been using. What has everyone's experience been with these new technologies?"
)