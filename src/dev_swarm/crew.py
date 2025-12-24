from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, FileReadTool, DirectoryReadTool
from pydantic import BaseModel, Field
from .tools.system_tools import AdvancedFileWriterTool
from .tools.qa_tools import CodeExecutionTool
from typing import List, Optional
import os

# --- Structured Output Models ---
class AgentSchema(BaseModel):
    role: str
    goal: str
    backstory: str
    tools: List[str] = Field(default_factory=list)

class TaskSchema(BaseModel):
    name: str
    description: str
    expected_output: str
    assigned_agent_role: str

class ProjectBlueprint(BaseModel):
    project_name: str
    stack: str
    required_agents: List[AgentSchema]
    dynamic_tasks: List[TaskSchema]

@CrewBase
class DevSwarmCrew():
    """DevSwarm: l'agence autonome auto-configurable"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    
    # Utilisation des variables d'environnement pour les modèles
    high_reasoning_model = os.getenv("MANAGER_MODEL", "gpt-4o")
    standard_model = os.getenv("MODEL", "gpt-4o-mini")

    def hire_dynamic_agent(self, schema: AgentSchema) -> Agent:
        """Instancie un agent spécialisé dynamiquement."""
        tool_map = {
            "FileWriterTool": AdvancedFileWriterTool(),
            "FileReadTool": FileReadTool(),
            "SerperDevTool": SerperDevTool(),
            "DirectoryReadTool": DirectoryReadTool(directory_path='project_output/src'),
            "CodeExecutionTool": CodeExecutionTool()
        }
        assigned_tools = [tool_map[t] for t in schema.tools if t in tool_map]
        
        return Agent(
            role=schema.role,
            goal=schema.goal,
            backstory=schema.backstory,
            tools=assigned_tools,
            llm=self.standard_model,
            verbose=True
        )

    @agent
    def architect(self) -> Agent:
        return Agent(
            config=self.agents_config['architect'], 
            tools=[SerperDevTool(), AdvancedFileWriterTool()],
            llm=self.high_reasoning_model,
            verbose=True
        )

    @task
    def discovery_design_task(self) -> Task:
        return Task(
            config=self.tasks_config['discovery_design_task'],
            output_pydantic=ProjectBlueprint
        )

    @crew
    def crew(self) -> Crew:
        # On ne passe que ce qui est défini ici
        return Crew(
            agents=[self.architect()],
            tasks=[self.discovery_design_task()],
            process=Process.sequential,
            verbose=True
        )
