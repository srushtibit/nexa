import pandas as pd
from typing import Dict, Any

class TicketLookupTool:
    """A tool for looking up specific tickets by their ID from a CSV file."""
    def __init__(self, csv_path: str):
        """
        Initializes the tool by loading ticket data from a CSV file.
        Args:
            csv_path (str): The file path to the nexacorp_tickets.csv file.
        """
        try:
            # Load the CSV file into a pandas DataFrame
            self.df = pd.read_csv(csv_path, encoding='utf-8')
            # Set the 'Complaint ID' as the index for faster lookups
            self.df.set_index('Complaint ID', inplace=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"The ticket data file was not found at {csv_path}")
        except KeyError:
            raise KeyError("The CSV file must contain a 'Complaint ID' column.")

    def process_request(self, ticket_id: str) -> Dict[str, Any]:
        """
        Looks up a ticket by its ID and returns its details as a dictionary.
        Args:
            ticket_id (str): The ID of the ticket to look up (e.g., "NCX-12345").
        Returns:
            A dictionary containing the ticket's data or an error message.
        """
        try:
            # Use .loc for fast index-based lookup
            ticket_data = self.df.loc[ticket_id.strip()]
            # Convert the resulting series to a dictionary
            return ticket_data.to_dict()
        except KeyError:
            return {"error": f"Ticket ID '{ticket_id}' not found."}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {e}"}

