import os
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
load_dotenv()



llm = ChatOpenAI(model="gpt-4", temperature=0.7)


search_tool = SerperDevTool()


research_analyst = Agent(
    role="Research Analyst",
    goal="Understand the current AI safety landscape",
    backstory="You're gathering insights to guide the launch of a new AI safety startup.",
    tools=[search_tool],
    llm=llm,
    verbose=True
)


market_strategist = Agent(
    role="Market Strategist",
    goal="Translate research into a go-to-market strategy",
    backstory="You're helping define market entry and positioning based on analyst research.",
    tools=[],
    llm=llm,
    verbose=True
)


executive_advisor = Agent(
    role="Executive Advisor",
    goal="Review and finalize the business insights for leadership",
    backstory="You're a senior advisor evaluating startup reports and providing actionable guidance.",
    tools=[],
    llm=llm,
    verbose=True
)


task1 = Task(
    description="Use web search to summarize 3 current AI safety risks or trends.",
    expected_output="Bullet list with at least 3 trends, including key players and challenges.",
    agent=research_analyst
)

task2 = Task(
    description="Turn the analyst research into a 2-paragraph strategic market entry report.",
    expected_output="Concise report with opportunity, market gaps, and positioning.",
    agent=market_strategist,
    depends_on=[task1]
)

task3 = Task(
    description="Review the strategy and generate a 3-point executive summary with recommendations.",
    expected_output="An executive briefing outlining risks, strengths, and go/no-go decision factors.",
    agent=executive_advisor,
    depends_on=[task2]
)


crew = Crew(
    agents=[research_analyst, market_strategist, executive_advisor],
    tasks=[task1, task2, task3],
    verbose=True
)

result = crew.kickoff()
print("\n📋 Final Output:\n", result)
