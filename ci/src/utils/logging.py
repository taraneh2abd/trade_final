# ci/src/utils/logging.py
import logging
from colorama import init, Fore, Back, Style
from tabulate import tabulate
from typing import Optional

# Initialize colorama
init(autoreset=True)

def get_logger(name: str = "project_ci", level: int = logging.INFO, logfile: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    
    # Color formatter for console
    class ColorFormatter(logging.Formatter):
        COLORS = {
            'INFO': Fore.CYAN,
            'DEBUG': Fore.BLUE,
            'WARNING': Fore.YELLOW,
            'ERROR': Fore.RED,
            'CRITICAL': Back.RED + Fore.WHITE
        }
        
        def format(self, record):
            color = self.COLORS.get(record.levelname, Fore.WHITE)
            record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
            record.name = f"{Fore.MAGENTA}{record.name}{Style.RESET_ALL}"
            return super().format(record)
    
    # Console handler with color
    console_handler = logging.StreamHandler()
    console_formatter = ColorFormatter(
        "[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (no color)
    if logfile:
        file_handler = logging.FileHandler(logfile, encoding="utf-8")
        file_formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


# Helper function for pretty tables
def print_results_table(results, title="RESULTS"):
    """Print a beautiful table of results"""
    print(f"\n{Fore.CYAN}{'═' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{title:^60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'═' * 60}{Style.RESET_ALL}")
    
    if not results:
        print(f"{Fore.YELLOW}No results to display{Style.RESET_ALL}")
        return
    
    # Prepare table data
    table_data = []
    for result in results:
        row = []
        for key, value in result.items():
            # Colorize based on content
            if key == 'assessment':
                if 'good' in str(value).lower():
                    row.append(f"{Fore.GREEN}{value}{Style.RESET_ALL}")
                elif 'poor' in str(value).lower():
                    row.append(f"{Fore.RED}{value}{Style.RESET_ALL}")
                elif 'acceptable' in str(value).lower():
                    row.append(f"{Fore.YELLOW}{value}{Style.RESET_ALL}")
                else:
                    row.append(str(value))
            elif key == 'fitness':
                row.append(f"{Fore.BLUE}{value:.6f}{Style.RESET_ALL}")
            elif key == 'time':
                row.append(f"{Fore.MAGENTA}{value:.2f}s{Style.RESET_ALL}")
            elif key == 'method':
                row.append(f"{Fore.CYAN}{value}{Style.RESET_ALL}")
            else:
                row.append(str(value))
        table_data.append(row)
    
    # Print table using simple formatting
    headers = list(results[0].keys())
    
    # Calculate column widths
    col_widths = []
    for i, header in enumerate(headers):
        max_width = len(header)
        for row in table_data:
            max_width = max(max_width, len(str(row[i])))
        col_widths.append(max_width + 2)
    
    # Print header
    header_line = "┌" + "┬".join("─" * w for w in col_widths) + "┐"
    print(f"{Fore.WHITE}{header_line}{Style.RESET_ALL}")
    
    header_cells = []
    for i, header in enumerate(headers):
        header_cells.append(f"{Fore.YELLOW}{header:^{col_widths[i]}}{Style.RESET_ALL}")
    print("│" + "│".join(header_cells) + "│")
    
    # Print separator
    sep_line = "├" + "┼".join("─" * w for w in col_widths) + "┤"
    print(f"{Fore.WHITE}{sep_line}{Style.RESET_ALL}")
    
    # Print rows
    for row in table_data:
        row_cells = []
        for i, cell in enumerate(row):
            row_cells.append(f"{cell:<{col_widths[i]}}")
        print("│" + "│".join(row_cells) + "│")
    
    # Print footer
    footer_line = "└" + "┴".join("─" * w for w in col_widths) + "┘"
    print(f"{Fore.WHITE}{footer_line}{Style.RESET_ALL}")


def print_experiment_header(problem_name, run_num, total_runs):
    """Print experiment header"""
    print(f"\n{Fore.GREEN}{'─' * 60}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}🔬 Experiment {run_num}/{total_runs}: {problem_name}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'─' * 60}{Style.RESET_ALL}")