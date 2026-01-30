"""
Baseline prediction system using Pydantic AI for patient classification.
"""

from pathlib import Path
from pydantic_ai import Agent

from models import PatientPrediction
from utils import get_patient_notes


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


async def predict_patient(
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


if __name__ == "__main__":
    import asyncio
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    
    console = Console()
    
    async def main():
        # Example: Predict for Patient 001
        # Note: Requires ANTHROPIC_API_KEY environment variable
        try:
            console.print("[bold cyan]Running prediction for Patient 001...[/bold cyan]")
            prediction = await predict_patient("001")
            
            # Create results table
            table = Table(title="Patient 001 Predictions", show_header=True, header_style="bold magenta")
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
    
    asyncio.run(main())
