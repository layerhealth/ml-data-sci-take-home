"""
Generic utility script to run predictions across all patients and save to CSV.
Takes a prediction function and applies it to all patients asynchronously.
"""

import asyncio
from pathlib import Path
from typing import Callable, Dict, Awaitable
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from baseline import predict_patient
from models import PatientPrediction


console = Console()


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
    predict_fn: Callable[[str, Path, str, float], Awaitable[PatientPrediction]] = predict_patient,
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
    
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Total patients: {len(all_results)}")
    console.print(f"  Successful: [green]{successful}[/green]")
    if failed > 0:
        console.print(f"  Failed: [red]{failed}[/red]")


if __name__ == "__main__":
    import typer
    
    app = typer.Typer(help="Run predictions on all patients using the baseline prediction function")
    
    @app.command()
    def main(
        output: str = typer.Option(
            "predictions.csv",
            "--output",
            "-o",
            help="Output CSV file path"
        ),
        batch_size: int = typer.Option(
            10,
            "--batch-size",
            "-b",
            help="Number of concurrent predictions"
        ),
    ):
        """Run predictions on all patients and save to CSV."""
        asyncio.run(predict_all_patients(predict_fn=predict_patient, output_file=Path(output), batch_size=batch_size))
    
    app()
