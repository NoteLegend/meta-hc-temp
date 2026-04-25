import uvicorn
import sys
import os
import sqlite3
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# Force the root directory into the python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import env
from models import FinAuditAction, FinAuditObservation

app = FastAPI(title="FinAudit Environment API")
# Instantiate the NEW environment
env_instance = env.FinAuditEnv()

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """Generates a live HTML view of the current database."""
    db_path = "working.db" 
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🗄️ FinAudit - Live View</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f9; color: #333; padding: 2rem; max-width: 1200px; margin: 0 auto; }
            h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            .table-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 2rem; }
            h2 { color: #e74c3c; margin-top: 0; }
            table { width: 100%; border-collapse: collapse; margin-top: 10px; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f8f9fa; font-weight: 600; color: #2c3e50; }
            tr:hover { background-color: #f1f2f6; }
            .error { color: red; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>🗄️ FinAudit Environment - Live View</h1>
        <p>This dashboard shows the real-time state of the agent's workspace database. Refresh the page to see changes.</p>
    """

    if not os.path.exists(db_path):
        html_content += f"<p class='error'>Waiting for environment to initialize... ({db_path} not found).</p>"
    else:
        try:
            with sqlite3.connect(db_path) as conn:
                c = conn.cursor()
                c.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%';")
                db_objects = c.fetchall()
                
                if not db_objects:
                    html_content += "<p>Database is empty.</p>"
                
                for obj_name, obj_type in db_objects:
                    html_content += f"<div class='table-container'><h2>{obj_name} ({obj_type})</h2>"
                    c.execute(f"SELECT * FROM {obj_name} LIMIT 50")
                    rows = c.fetchall()
                    
                    if not rows:
                        html_content += "<p><em>No data</em></p></div>"
                        continue
                        
                    cols = [description[0] for description in c.description]
                    html_content += "<table><thead><tr>"
                    for col in cols: html_content += f"<th>{col}</th>"
                    html_content += "</tr></thead><tbody>"
                    
                    for row in rows:
                        html_content += "<tr>"
                        for val in row: html_content += f"<td>{str(val)}</td>"
                        html_content += "</tr>"
                        
                    html_content += "</tbody></table></div>"
                    
        except Exception as e:
            html_content += f"<p class='error'>Error reading database: {str(e)}</p>"

    html_content += "</body></html>"
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/reset", response_model=FinAuditObservation)
def reset_env(task_name: str = None):
    """Resets the environment and returns the initial observation."""
    obs_dict = env_instance.reset(task_name)
    return obs_dict

@app.post("/step")
def step_env(action: FinAuditAction):
    """Executes a single step in the environment."""
    obs_dict, reward, done = env_instance.step(action.model_dump())
    return {
        "observation": obs_dict,
        "reward": reward,
        "done": done
    }

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()