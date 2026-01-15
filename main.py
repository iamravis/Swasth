import sys
import logging
import requests
import argparse
import shutil
import time
import subprocess
import yaml
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.logging import RichHandler
from rich.theme import Theme
from src.pipeline import UnifiedPipeline
from src.generate_qa import QualityGenerator

# Setup Professional Theme
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "phase": "bold magenta"
})

console = Console(theme=custom_theme)

# Configure rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)]
)
logger = logging.getLogger("rich")

# Suppress verbose logs from dependencies
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("synthetic_data_kit").setLevel(logging.WARNING)
logging.getLogger("docling").setLevel(logging.WARNING)

def print_banner():
    banner = """
    🧬 SWASTH: MEDICAL SFT DATA PIPELINE
    ====================================
    """
    console.print(Panel(banner, style="phase", expand=False))
    console.print("\n") # Space after banner

def check_ollama(model_name=None):
    """Check if Ollama is running and specifically if the model is available."""
    try:
        # 1. Basic Ping
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code != 200:
            return False
            
        # 2. Model Check (Optional but recommended)
        if model_name:
            tags = response.json().get("models", [])
            # Look for exact match or name match (Ollama tags can be complex)
            model_exists = any(model_name in m.get("name", "") for m in tags)
            if not model_exists:
                # If model not found, we don't return False here because 
                # Ollama might pull it on the fly, but we can log a warning.
                pass
        return True
    except Exception:
        pass
    return False

def is_ollama_process_running():
    """Check if any Ollama process is running."""
    try:
        output = subprocess.check_output(["ps", "aux"], text=True)
        return "Ollama.app" in output or "ollama serve" in output or "ollama" in output.lower()
    except:
        return False

def ensure_ollama(model_name=None):
    """Ensure Ollama is running, attempting to start it if necessary."""
    if check_ollama(model_name):
        console.print("[success]✅ Ollama is online and reachable.[/success]")
        return True

    if is_ollama_process_running():
        console.print("[warning]ℹ️  Ollama process detected but API not responsive. Waiting longer...[/warning]")
    else:
        console.print("[warning]⚠️  Ollama is not running. Attempting to start...[/warning]")
        try:
            if sys.platform == "darwin":
                # Try starting the app directly
                subprocess.Popen(["open", "/Applications/Ollama.app"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            console.print(f"[error]❌ Failed to launch Ollama: {e}[/error]")
            return False
    
    # Wait for readiness
    with console.status("[bold yellow]Initializing Ollama engine...", spinner="bouncingBar"):
        max_retries = 45 # Increased for safety
        for i in range(max_retries):
            time.sleep(1)
            if check_ollama(model_name):
                console.print(f"[success]✅ Ollama is ready (Started in {i+1}s)[/success]")
                return True
                
    console.print("[error]❌ Ollama API did not become ready after 45s.[/error]")
    console.print("[info]Tip: Try starting Ollama manually and then re-run this script.[/info]")
    return False

def ingest_file(file_path):
    """Copy a single file into the data/raw directory."""
    src = Path(file_path)
    if not src.exists():
        console.print(f"[error]❌ File not found:[/error] {file_path}")
        sys.exit(1)
        
    dest_dir = Path("data/raw")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    
    if dest.exists():
        console.print(f"[warning]ℹ️  File {src.name} already exists in data/raw.[/warning]")
    else:
        shutil.copy2(src, dest)
        console.print(f"[success]📥 Ingested {src.name} to data/raw/[/success]")
    console.print("\n") # Space after ingestion

def run_all():
    parser = argparse.ArgumentParser(description="Unified Medical SFT Pipeline")
    parser.add_argument("--file", "-f", help="Path to a new PDF/HTML/DOCX to ingest and process")
    parser.add_argument("--config", "-c", default="config/sft_config.yaml", help="Path to configuration file")
    args = parser.parse_args()

    # Load config for model name verification
    model_name = None
    try:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
            model_name = cfg.get("api-endpoint", {}).get("model")
    except Exception:
        pass

    print_banner()
    
    # 0. Ingestion (Optional)
    if args.file:
        console.print("[phase]━━━━ PHASE 0: FILE INGESTION ━━━━[/phase]")
        ingest_file(args.file)

    # 1. Extraction & Structuring
    console.print("[phase]━━━━ PHASE 1: EXTRACTION & STRUCTURING ━━━━[/phase]")
    with console.status("[bold cyan]Processing raw guidelines...", spinner="dots"):
        pipeline = UnifiedPipeline()
        pipeline.process()
    console.print("\n") # Space after Phase 1
    
    # 2. Service Health Check (Self-Healing)
    console.print("[phase]━━━━ PHASE 2: SERVICE HEALTH CHECK (SELF-HEALING) ━━━━[/phase]")
    if not ensure_ollama(model_name):
        console.print("[warning]⚠️  Skipping Stage 3 (QA Generation) due to backend unavailability.[/warning]")
        sys.exit(1)
    console.print("\n") # Space after Phase 2
        
    # 3. QA Generation
    console.print("[phase]━━━━ PHASE 3: HIGH-FIDELITY QA GENERATION ━━━━[/phase]")
    gen = QualityGenerator(config_path=args.config)
    gen.generate_all()
    
    console.print("\n[success]✨ Pipeline Execution Complete![/success]\n")

if __name__ == "__main__":
    run_all()
