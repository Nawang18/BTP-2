import sys

try:
    from fpdf import FPDF
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fpdf2"])
    from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, 'BTP Mid-Term Report', 0, 1, 'C')
        self.set_font('Helvetica', '', 12)
        self.cell(0, 10, 'Multi-Robot Task Orchestration for Construction Assembly', 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)

    def chapter_body(self, body):
        self.set_font('Helvetica', '', 11)
        self.multi_cell(0, 6, body)
        self.ln(5)

pdf = PDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)

content = {
    "1. Project Overview": (
        "Total Duration: 4 Months\n"
        "Time Elapsed: 2 Months (August - September)\n"
        "Time Remaining: 2 Months\n\n"
        "Final Goal: To develop a system where multiple robots (e.g., Franka Panda arm, shuttle, mortar dispenser) "
        "work collaboratively, orchestrated by an LLM (Large Language Model) brain, to construct a brick wall. "
        "Key objectives include precise placement, task orchestration, and seamless multi-robot coordination."
    ),
    "2. Progress Till Now": (
        "Based on the current codebase, significant milestones have been achieved in the first two months:\n\n"
        "- Simulation Environment: A fully functional PyBullet simulation environment has been established, "
        "complete with bricks, pallet stacks, and a pedestal.\n"
        "- Robot Control & Kinematics: Panda Arm implemented with grasp-error compensation ensuring 1-2mm precision. "
        "Shuttle and Mortar Bot developed with collision-safe routing. Robot behaviors are implemented as Python "
        "generators allowing concurrent step-by-step execution.\n"
        "- Task Orchestration (DAG): Developed a task dependency graph and deterministic scheduler that correctly "
        "sequences transport, mortar application, and placement tasks.\n"
        "- Vision Integration: Integrated 3D Computer Vision (OpenCV RGB-D) for overhead camera detection, "
        "allowing the system to grasp tilted bricks and perform reachability checks.\n"
        "- LLM Brain Integration: Initiated development of a 3-Layer LLM Brain (using Gemini models) for strategic "
        "planning and tactical supervision."
    ),
    "3. Current Challenges & Problems": (
        "- Simplified Physics Interactions: The shuttle currently 'picks up' and 'drops' bricks by teleporting them "
        "between the dock and deck. Mortar is simulated as a static slab rather than a fluid extrusion.\n"
        "- Perception Assumptions: In some headless modes, placement precision assumes perfect perception by reading "
        "directly from the simulation state, bypassing real-world vision noise.\n"
        "- Workspace Constraints: The robotic arm operates in a fixed workspace. Constructing larger walls requires "
        "dynamic base placement or a mobile base for the arm.\n"
        "- Error Recovery Robustness: The closed-loop error recovery orchestrated by the LLM Tactical Supervisor is "
        "still under development and needs stress-testing."
    ),
    "4. Future Goals (Next 2 Months)": (
        "- Refine LLM Orchestration: Fully transition the mock LLM front-end to a robust LLM brain capable of handling "
        "complex, ambiguous natural language commands.\n"
        "- Enhance Tactical Supervisor: Improve the LLM's ability to monitor the world state and autonomously trigger "
        "recovery tools (retry, skip, adjust) without human intervention.\n"
        "- Improve Physics Realism: Replace teleportation mechanisms with realistic kinematic grasping for the shuttle.\n"
        "- Comprehensive Benchmarking: Run extensive evaluations to gather metrics on placement accuracy, timeline "
        "efficiency, and failure rates under various edge cases.\n"
        "- Final Project Deliverables: Finalize documentation, prepare final viva defense, and ensure all claims are "
        "backed by simulation data."
    )
}

for title, body in content.items():
    pdf.chapter_title(title)
    pdf.chapter_body(body)

pdf.output('BTP_MidTerm_Report.pdf')
print("PDF report created successfully at BTP_MidTerm_Report.pdf")
