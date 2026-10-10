"""
Prediction system for patient classification using Pydantic AI.
Supports both single-patient and batch predictions.
"""

import asyncio
from pathlib import Path
from typing import Callable, Dict, Awaitable, Optional

import pandas as pd
from pydantic_ai import Agent
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
import typer

from models import PatientPrediction
from utils import get_patient_notes


console = Console()


# System prompt for clinical classification
SYSTEM_PROMPT = """You are an expert clinical data extractor specializing in EHR analysis.

Extract and classify the following information from the patient's clinical notes:

1. **diabetes**: Patient's diabetes status - must be exactly one of:
   - "No Diabetes"
   - "Prediabetes" 
   - "Type 1 Diabetes"
   - "Type 2 Diabetes"

2. **smoking**: Patient's smoking status - must be exactly one of:
   - "Smoker" (current cigarette use OR quit less than 1 year ago)
   - "Not Smoker" (never smoked cigarettes, quit more than 1 year ago, or only non-cigarette tobacco products)

3. **cancer**: Patient's cancer status - must be exactly one of:
   - "No Cancer/Benign"
   - "Indeterminate"
   - "Active"
   - "Remission"

4. **reasoning**: Provide your clinical reasoning for each classification

5. **supporting_quotes**: List specific quotes from the notes that support your classifications"""


async def baseline_predict_patient(
    patient_id: str, 
    data_dir: Path = Path("../data"),
    model_name: str = "anthropic:claude-haiku-4-5-20251001",
    temperature: float = 1.0
) -> PatientPrediction:
    """
    Generate predictions for a patient using Pydantic AI.
    
    Args:
        patient_id: Patient identifier (e.g., "001")
        data_dir: Path to data directory
        model_name: Model to use for prediction (default: claude-3-5-haiku-20241022)
        temperature: Sampling temperature (0.0-1.0, default: 1.0)
    
    Returns:
        PatientPrediction with diabetes, smoking, cancer predictions and reasoning
    
    Raises:
        FileNotFoundError: If patient notes not found
    
    Note:
        Requires ANTHROPIC_API_KEY environment variable to be set
    """
    # Get list of notes from utils
    notes = get_patient_notes(patient_id, data_dir)
    
    # Concatenate notes with clear separators
    concatenated_notes = "\n\n---NOTE---\n\n".join(notes)
    
    # Create Pydantic AI agent with structured output
    agent = Agent(
        model_name,
        system_prompt=SYSTEM_PROMPT,
        output_type=PatientPrediction,
    )
    
    # Run prediction with temperature
    result = await agent.run(
        f"Extract the diabetes, smoking, and cancer classifications from these clinical notes:\n\n{concatenated_notes}",
        model_settings={"temperature": temperature}
    )
    return result.output


async def predict_patient_async(
    patient_id: str,
    predict_fn: Callable[[str, Path, str, float], Awaitable[PatientPrediction]],
    data_dir: Path,
    model_name: str,
    temperature: float
) -> Dict:
    """
    Run async prediction function for a single patient.
    
    Args:
        patient_id: Patient identifier (e.g., "001")
        predict_fn: Async prediction function to call
        data_dir: Path to data directory
        model_name: Model name to pass to prediction function
        temperature: Temperature for sampling
    
    Returns:
        Dictionary with patient_id and prediction results
    """
    try:
        # Call the async prediction function directly
        prediction = await predict_fn(patient_id, data_dir, model_name, temperature)
        
        return {
            "patient_id": patient_id,
            "diabetes": prediction.diabetes,
            "smoking": prediction.smoking,
            "cancer": prediction.cancer,
            "reasoning": prediction.reasoning,
            "supporting_quotes": " | ".join(prediction.supporting_quotes) if prediction.supporting_quotes else ""
        }
    except Exception as e:
        console.print(f"[red]Error processing patient {patient_id}: {e}[/red]")
        return {
            "patient_id": patient_id,
            "diabetes": "ERROR",
            "smoking": "ERROR",
            "cancer": "ERROR",
            "reasoning": f"Error: {str(e)}",
            "supporting_quotes": ""
        }


async def predict_all_patients(
    predict_fn: Callable[[str, Path, str, float], Awaitable[PatientPrediction]] = baseline_predict_patient,
    model_name: str = "anthropic:claude-haiku-4-5-20251001",
    data_dir: Path = Path("../data"),
    output_file: Path = Path("predictions.csv"),
    batch_size: int = 10,
    temperature: float = 1.0
) -> None:
    """
    Run predictions on all patients asynchronously and save to CSV.
    
    Args:
        predict_fn: Prediction function to use (takes patient_id, data_dir, model_name, temperature)
        model_name: Model to use for predictions
        data_dir: Path to data directory
        output_file: Path to output CSV file
        batch_size: Number of concurrent predictions to run
        temperature: Sampling temperature (0.0-1.0)
    """
    # Find all patient directories
    patient_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("Pt_")])
    patient_ids = [d.name.replace("Pt_", "") for d in patient_dirs]
    
    console.print(f"[bold cyan]Found {len(patient_ids)} patients to process[/bold cyan]")
    console.print(f"[cyan]Using model: {model_name}[/cyan]")
    console.print(f"[cyan]Temperature: {temperature}[/cyan]")
    console.print(f"[cyan]Batch size: {batch_size}[/cyan]\n")
    
    # Process patients in batches with progress bar
    all_results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Processing patients...", total=len(patient_ids))
        
        for i in range(0, len(patient_ids), batch_size):
            batch = patient_ids[i:i + batch_size]
            
            # Run batch predictions concurrently
            tasks = [
                predict_patient_async(patient_id, predict_fn, data_dir, model_name, temperature) 
                for patient_id in batch
            ]
            batch_results = await asyncio.gather(*tasks)
            
            all_results.extend(batch_results)
            progress.update(task, advance=len(batch))
    
    # Write results to CSV using pandas
    console.print(f"\n[bold green]Writing results to {output_file}[/bold green]")
    
    df = pd.DataFrame(all_results)
    df = df[['patient_id', 'diabetes', 'smoking', 'cancer', 'reasoning', 'supporting_quotes']]
    df.to_csv(output_file, index=False)
    
    console.print(f"[bold green]✓ Predictions saved to {output_file}[/bold green]")
    
    # Print summary statistics
    successful = sum(1 for r in all_results if r['diabetes'] != 'ERROR')
    failed = len(all_results) - successful
    
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Total patients: {len(all_results)}")
    console.print(f"  Successful: [green]{successful}[/green]")
    if failed > 0:
        console.print(f"  Failed: [red]{failed}[/red]")


async def predict_single_patient(patient_id: str) -> None:
    """
    Run prediction for a single patient and display results.
    
    Args:
        patient_id: Patient identifier (e.g., "001")
    """
    try:
        console.print(f"[bold cyan]Running prediction for Patient {patient_id}...[/bold cyan]")
        prediction = await baseline_predict_patient(patient_id)
        
        # Create results table
        table = Table(title=f"Patient {patient_id} Predictions", show_header=True, header_style="bold magenta")
        table.add_column("Category", style="cyan", width=20)
        table.add_column("Classification", style="green")
        
        table.add_row("Diabetes", prediction.diabetes)
        table.add_row("Smoking", prediction.smoking)
        table.add_row("Cancer", prediction.cancer)
        
        console.print(table)
        
        # Print reasoning
        console.print(Panel(prediction.reasoning, title="[bold]Clinical Reasoning[/bold]", border_style="blue"))
        
        # Print supporting quotes
        if prediction.supporting_quotes:
            console.print("\n[bold yellow]Supporting Quotes:[/bold yellow]")
            for i, quote in enumerate(prediction.supporting_quotes, 1):
                console.print(f"  [dim]{i}.[/dim] {quote}")
        else:
            console.print("\n[dim]No supporting quotes provided.[/dim]")
                
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    app = typer.Typer(help="Run predictions on patients using the baseline prediction function")
    
    @app.command()
    def main(
        patient: Optional[str] = typer.Option(
            None,
            "--patient",
            "-p",
            help="Single patient ID to predict (e.g., '001'). If not provided, runs batch predictions on all patients."
        ),
        output: str = typer.Option(
            "predictions.csv",
            "--output",
            "-o",
            help="Output CSV file path (only used for batch predictions)"
        ),
        batch_size: int = typer.Option(
            10,
            "--batch-size",
            "-b",
            help="Number of concurrent predictions (only used for batch predictions)"
        ),
    ):
        """Run predictions on patients and save to CSV."""
        if patient:
            # Single patient mode
            asyncio.run(predict_single_patient(patient))
        else:
            # Batch mode
            asyncio.run(predict_all_patients(predict_fn=baseline_predict_patient, output_file=Path(output), batch_size=batch_size))
    
    app()
