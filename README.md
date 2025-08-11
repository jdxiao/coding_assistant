## Group 24 ECE1508 Course Project

### Topic B-1: Educational Code Generation using LLMs with Self-Refinement

Overview
This project explores using Large Language Models (LLMs) for educational code generation, enhanced by a self-refinement process.
The pipeline evaluates different prompt styles and measures their impact on code quality, clarity, and documentation.

Features
Multiple prompt variations (v1_basic, v2_docstring, v3_plan_then_code)
Automated code generation with StarCoder
Self-refinement loop using execution feedback
Metrics: pass rate, comment density, docstring presence
Visualizations comparing prompt performance

File Structure
code (18).ipynb          # Main notebook: generation, refinement, metrics, visualization
self_refinement (2).py   # Self-refinement logic as a standalone script
archive/                 # Deprecated or earlier versions
README.md                # Project documentation

Quickstart
Install dependencies:
pip install -r requirements.txt

Run the main notebook
Open code (23).ipynb in Jupyter and run all cells.

Or run the refinement script
python self_refinement (3).py

Here is an Example Metrics Output  
| Prompt           | gen_comment_density | gen_has_docstring | gen_pass_rate | ref_comment_density | ref_has_docstring | ref_pass_rate |
|------------------|---------------------|-------------------|---------------|---------------------|-------------------|---------------|
| v1_basic         | 0.520               | False             | 0.0           | 0.520               | False             | 0.0           |
| v2_docstring     | 0.613               | True              | 0.0           | 0.559               | True              | 0.0           |
| v3_plan_then_code| 0.000               | False             | 0.0           | 0.000               | False             | 0.0           |


Notes
Running the notebook on large datasets may take several hours.
Predefined small datasets are available for faster demo runs.