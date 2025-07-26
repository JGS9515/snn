import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

class GanttChartGenerator:
    def __init__(self, figsize=(12, 8)):
        """
        Initialize the Gantt Chart Generator
        
        Args:
            figsize (tuple): Figure size for the plot
        """
        self.figsize = figsize
        self.fig = None
        self.ax = None
        self.tasks = []
        self.colors = ['#FFD700', '#87CEEB', '#FFA07A', '#98FB98', '#DDA0DD', 
                      '#F0E68C', '#FFB6C1', '#B0E0E6', '#FFDAB9', '#E6E6FA']
        
    def add_task(self, task_name, start_date, duration, category=None, dependencies=None):
        """
        Add a task to the Gantt chart
        
        Args:
            task_name (str): Name of the task
            start_date (datetime or str): Start date of the task
            duration (int): Duration in days
            category (str, optional): Category for color coding
            dependencies (list, optional): List of task names this task depends on
        """
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        
        end_date = start_date + timedelta(days=duration)
        
        self.tasks.append({
            'task_name': task_name,            'start_date': start_date,
            'end_date': end_date,
            'duration': duration,
            'category': category,
            'dependencies': dependencies or []
        })
    
    def create_sample_project(self):
        """
        Create a sample project similar to the one in the image
        Tasks are added in the order they should appear (top to bottom)
        """
        # Clear existing tasks
        self.tasks = []
        
        # Main phases in logical order from the image
        self.add_task("Búsqueda de información sobre tecnología necesaria", "2024-01-01", 30, "research")
        
        # Sensor technology block
        self.add_task("Tecnología de sensores", "2024-01-15", 45, "sensors")
        self.add_task("    Aplicación web", "2024-02-01", 25, "web")
        self.add_task("    Servicios REST", "2024-02-15", 20, "api")
        self.add_task("    Aplicación Android", "2024-03-01", 30, "mobile")
        self.add_task("    Búsqueda de bibliografía", "2024-03-15", 15, "research")
        self.add_task("    Búsqueda de bibliografía", "2024-04-01", 15, "research")
        
        # Hardware development
        self.add_task("Desarrollo de los sensores (hardware)", "2024-04-15", 40, "hardware")
        self.add_task("Desarrollo de los sensores (hardware)", "2024-05-01", 35, "hardware")
        
        # Software development block
        self.add_task("Desarrollo software", "2024-05-15", 60, "software")
        self.add_task("    Sensores", "2024-06-01", 30, "sensors")
        self.add_task("    Diseño", "2024-06-15", 25, "design")
        self.add_task("    Implementación", "2024-07-01", 35, "implementation")
        
        # Web application block
        self.add_task("Aplicación web", "2024-07-15", 45, "web")
        self.add_task("    Diseño", "2024-08-01", 20, "design")
        self.add_task("    Implementación", "2024-08-15", 30, "implementation")
        self.add_task("    Implementación de servicios REST", "2024-09-01", 25, "api")
        
        # Mobile application block
        self.add_task("Aplicación Android", "2024-09-15", 40, "mobile")
        self.add_task("    Diseño", "2024-10-01", 15, "design")
        self.add_task("    Implementación", "2024-10-15", 25, "implementation")
        
        # Testing and deployment
        self.add_task("Pruebas software", "2024-11-01", 20, "testing")
        self.add_task("Corrección de posibles errores", "2024-11-15", 15, "testing")
        self.add_task("Instalación de software", "2024-12-01", 10, "deployment")
        self.add_task("Preparación del servidor", "2024-12-05", 12, "deployment")
        self.add_task("Despliegue de aplicaciones", "2024-12-10", 15, "deployment")
        
        # Documentation
        self.add_task("Realización de la documentación", "2024-12-15", 30, "documentation")
        self.add_task("    Memoria", "2024-12-20", 20, "documentation")
        self.add_task("    Manuales", "2024-12-25", 15, "documentation")
    
    def get_color_for_category(self, category):
        """
        Get color for a specific category
        """
        category_colors = {
            'research': '#FFD700',      # Gold
            'sensors': '#87CEEB',       # Sky Blue
            'web': '#FFA07A',          # Light Salmon
            'api': '#98FB98',          # Pale Green
            'mobile': '#DDA0DD',       # Plum
            'hardware': '#F0E68C',     # Khaki
            'software': '#FFB6C1',     # Light Pink
            'design': '#B0E0E6',       # Powder Blue
            'implementation': '#FFDAB9', # Peach Puff
            'testing': '#E6E6FA',      # Lavender
            'deployment': '#F5DEB3',   # Wheat
            'documentation': '#D3D3D3'  # Light Gray
        }
        return category_colors.get(category, '#CCCCCC')
    
    def create_gantt_chart(self, title="Diagrama de Gantt", save_path=None, time_unit="weeks"):
        """
        Create and display the Gantt chart
        
        Args:
            title (str): Title of the chart
            save_path (str, optional): Path to save the chart
            time_unit (str): "weeks" or "months" for time axis display
        """
        if not self.tasks:
            print("No tasks added. Use add_task() or create_sample_project() first.")
            return
        
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        
        # Keep tasks in the order they were added (proper Gantt chart order)
        # Tasks should be displayed top to bottom in logical project order
        ordered_tasks = self.tasks.copy()
        
        # Get date range
        min_date = min(task['start_date'] for task in ordered_tasks)
        max_date = max(task['end_date'] for task in ordered_tasks)
          # Create time axis
        total_days = (max_date - min_date).days        # Plot tasks from top to bottom
        for i, task in enumerate(ordered_tasks):
            # Calculate position: first task (i=0) gets highest y value
            y_pos = len(ordered_tasks) - 1 - i
            start_days = (task['start_date'] - min_date).days
            duration_days = task['duration']
            
            # Get color
            color = self.get_color_for_category(task['category'])
            
            # Create rectangle with EXACTLY the same height for all tasks
            rect = patches.Rectangle(
                (start_days, y_pos),     # x, y position  
                duration_days,           # width (duration)
                0.8,                    # height (IDENTICAL for all tasks)
                linewidth=1, 
                edgecolor='black', 
                facecolor=color,
                alpha=0.8
            )
            self.ax.add_patch(rect)
            
            # Debug print (can be removed later)
            if i < 3:  # Only print first 3 for debugging
                print(f"Task {i+1}: '{task['task_name'][:20]}...' -> y_pos={y_pos}, height=0.8")
        
        # Create labels in reverse order to match rectangle positions
        y_labels = [task['task_name'] for task in reversed(ordered_tasks)]
          # Configure axes with proper limits
        self.ax.set_xlim(0, total_days)
        self.ax.set_ylim(-0.5, len(ordered_tasks) - 0.5)
        
        # Set y-axis labels with proper alignment
        self.ax.set_yticks(range(len(ordered_tasks)))
        self.ax.set_yticklabels(y_labels, fontsize=8)
        
        # Ensure equal spacing and height for all tasks
        self.ax.set_aspect('auto')  # Allow proper scaling
        
        # Set x-axis based on time_unit
        if time_unit == "weeks":
            self._set_weekly_axis(total_days)
        else:  # months
            self._set_monthly_axis(min_date, max_date, total_days)
        
        # Add grid
        self.ax.grid(True, alpha=0.3)
        
        # Set title
        self.ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add legend
        self.add_legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to: {save_path}")
        
        # Show plot
        plt.show()
    
    def _set_weekly_axis(self, total_days):
        """Set x-axis to show weeks"""
        week_ticks = []
        week_labels = []
        
        # Calculate total weeks needed
        total_weeks = (total_days // 7) + (1 if total_days % 7 > 0 else 0)
        
        # Create week markers
        for week_num in range(total_weeks + 1):
            days_from_start = week_num * 7
            if days_from_start <= total_days:
                week_ticks.append(days_from_start)
                week_labels.append(f"Semana {week_num + 1}")
        
        self.ax.set_xticks(week_ticks)
        self.ax.set_xticklabels(week_labels, rotation=45, ha='right')
    
    def _set_monthly_axis(self, min_date, max_date, total_days):
        """Set x-axis to show months"""
        month_ticks = []
        month_labels = []
        current_date = min_date.replace(day=1)
        
        while current_date <= max_date:
            days_from_start = (current_date - min_date).days
            if days_from_start >= 0:
                month_ticks.append(days_from_start)
                month_labels.append(current_date.strftime('%b %Y'))
            
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
        
        self.ax.set_xticks(month_ticks)
        self.ax.set_xticklabels(month_labels, rotation=45, ha='right')
    
    def add_legend(self):
        """
        Add a legend to the chart
        """
        categories = set(task['category'] for task in self.tasks if task['category'])
        
        legend_elements = []
        for category in sorted(categories):
            color = self.get_color_for_category(category)
            legend_elements.append(
                patches.Patch(color=color, label=category.capitalize())
            )
        
        if legend_elements:
            self.ax.legend(
                handles=legend_elements, 
                loc='upper left', 
                bbox_to_anchor=(1.02, 1),
                fontsize=8
            )
    
    def export_to_csv(self, filename="gantt_data.csv"):
        """
        Export task data to CSV
        
        Args:
            filename (str): Name of the CSV file
        """
        if not self.tasks:
            print("No tasks to export.")
            return
        
        df = pd.DataFrame(self.tasks)
        df['start_date'] = df['start_date'].dt.strftime('%Y-%m-%d')
        df['end_date'] = df['end_date'].dt.strftime('%Y-%m-%d')
        
        df.to_csv(filename, index=False)
        print(f"Data exported to: {filename}")
    
    def load_from_csv(self, filename):
        """
        Load task data from CSV
        
        Args:
            filename (str): Name of the CSV file
        """
        try:
            df = pd.read_csv(filename)
            self.tasks = []
            
            for _, row in df.iterrows():
                self.add_task(
                    task_name=row['task_name'],
                    start_date=row['start_date'],
                    duration=row['duration'],
                    category=row.get('category', None)
                )
            
            print(f"Data loaded from: {filename}")
        except FileNotFoundError:
            print(f"File {filename} not found.")
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def get_project_statistics(self):
        """
        Get project statistics including total duration in weeks
        """
        if not self.tasks:
            return None
        
        min_date = min(task['start_date'] for task in self.tasks)
        max_date = max(task['end_date'] for task in self.tasks)
        
        total_days = (max_date - min_date).days
        total_weeks = (total_days // 7) + (1 if total_days % 7 > 0 else 0)
        
        stats = {
            'start_date': min_date,
            'end_date': max_date,
            'total_days': total_days,
            'total_weeks': total_weeks,
            'num_tasks': len(self.tasks)
        }
        
        return stats
    
    def print_project_summary(self):
        """
        Print a summary of the project timeline in weeks
        """
        stats = self.get_project_statistics()
        if not stats:
            print("No tasks in project.")
            return
        
        print(f"\n=== PROJECT SUMMARY ===")
        print(f"Total project duration: {stats['total_weeks']} weeks ({stats['total_days']} days)")
        print(f"Start date: {stats['start_date'].strftime('%Y-%m-%d')}")
        print(f"End date: {stats['end_date'].strftime('%Y-%m-%d')}")
        print(f"Number of tasks: {stats['num_tasks']}")
        print()
        
        # Show each task with its week timing
        min_date = stats['start_date']
        print("TASK TIMELINE (by weeks):")
        print("-" * 50)
        
        for i, task in enumerate(self.tasks, 1):
            start_week = ((task['start_date'] - min_date).days // 7) + 1
            end_week = ((task['end_date'] - min_date).days // 7) + 1
            duration_weeks = (task['duration'] // 7) + (1 if task['duration'] % 7 > 0 else 0)
            
            if start_week == end_week:
                week_range = f"Week {start_week}"
            else:
                week_range = f"Weeks {start_week}-{end_week}"
            
            print(f"{i:2d}. {task['task_name'][:40]:<40} | {week_range} ({duration_weeks} weeks)")

def main():
    """
    Main function to demonstrate the Gantt chart generator
    """
    # Create generator instance
    gantt = GanttChartGenerator(figsize=(14, 10))
      # Create sample project (similar to the image)
    gantt.create_sample_project()
    
    # Print project summary in weeks
    gantt.print_project_summary()
    
    # Generate and display the chart
    gantt.create_gantt_chart(
        title="Diagrama de Gantt - Proyecto de Desarrollo (Timeline in Weeks)",
        save_path="diagrama_gantt_weeks.png",
        time_unit="weeks"
    )
    
    # Export data to CSV
    gantt.export_to_csv("proyecto_gantt.csv")


if __name__ == "__main__":
    main()