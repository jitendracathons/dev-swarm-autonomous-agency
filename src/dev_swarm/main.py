#!/usr/bin/env python
import os
import shutil
from dev_swarm.crew import DevSwarmCrew
from crewai import Crew, Process, Task
from dotenv import load_dotenv

load_dotenv()

def run():
    # 1. CLEAN WORKSPACE
    if os.path.exists('project_output'):
        shutil.rmtree('project_output')
    os.makedirs('project_output/src', exist_ok=True)

    # Load Model strings from .env
    base_model = os.getenv("MODEL", "gpt-4o-mini")
    manager_model = os.getenv("MANAGER_MODEL", "gpt-4o")

    print(f"## Phase 1: Discovery using {manager_model}...")
    
    inputs = {
        'user_requirement': 'A high-speed real-time stock market tracker with SMS alerts.'
    }

    # 2. RUN THE ARCHITECT (Discovery Phase)
    crew_instance = DevSwarmCrew()
    discovery_result = crew_instance.crew().kickoff(inputs=inputs)
    blueprint = discovery_result.pydantic

    print(f"\n## Blueprint Created: {blueprint.project_name}")

    # --- SAFETY LIMITER & BUDGET CONTROL ---
    MAX_AGENTS = 2  # Only allow 2 extra specialists
    MAX_TASKS = 4   # Only allow 4 specialized tasks
    hiring_list = blueprint.required_agents[:MAX_AGENTS]
    task_list = blueprint.dynamic_tasks[:MAX_TASKS]

    # 3. DYNAMIC ASSEMBLY
    active_agents = { "Architect": crew_instance.architect() }
    for agent_schema in hiring_list:
        new_agent = crew_instance.hire_dynamic_agent(agent_schema)
        active_agents[agent_schema.role] = new_agent
        print(f"   -> Hired Specialist: {agent_schema.role}")

    # 4. DYNAMIC TASKING
    final_tasks = []
    for t_schema in task_list:
        executing_agent = active_agents.get(t_schema.assigned_agent_role, active_agents["Architect"])
        final_tasks.append(Task(
            description=t_schema.description,
            expected_output=t_schema.expected_output,
            agent=executing_agent
        ))
        print(f"   -> Created Task: {t_schema.name}")

    # 5. FINAL EXECUTION: The Evolved Swarm
    print(f"\n## Phase 2: Executing Swarm (Manager: {manager_model})...")
    
    
    evolved_crew = Crew(
        agents=list(active_agents.values()),
        tasks=final_tasks,
        process=Process.hierarchical, 
        manager_llm=manager_model, # High-reasoning manager
        verbose=True
    )

    final_output = evolved_crew.kickoff()
    
    print("\n\n########################")
    print("## MISSION ACCOMPLISHED ##")
    print("########################\n")
    print(f"Final Build Result: {final_output.raw}")

if __name__ == "__main__":
    run()