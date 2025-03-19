from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.wikipedia import WikipediaTools
from phi.tools.duckduckgo import DuckDuckGo

# Loading environment variables
from dotenv import load_dotenv
load_dotenv()

medicine_agent = Agent(
    name="Medicine Assistant",
    model=Groq(id="llama-3.2-1b-preview"),
    tools=[
        WikipediaTools(),
        DuckDuckGo()  # Useful for retrieving general information
    ],
