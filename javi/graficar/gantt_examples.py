"""
Example usage of the Gantt Chart Generator
This file demonstrates different ways to use the GanttChartGenerator class
"""

from diagrama_grant import GanttChartGenerator
from datetime import datetime

def create_software_project_example():
    """
    Create a software development project example
    """
    gantt = GanttChartGenerator(figsize=(12, 8))
    
    # Add tasks for a typical software project
    gantt.add_task("Requirements Analysis", "2024-01-01", 14, "analysis")
    gantt.add_task("System Design", "2024-01-15", 21, "design")
    gantt.add_task("Database Design", "2024-02-01", 10, "design")
    gantt.add_task("Frontend Development", "2024-02-15", 35, "frontend")
    gantt.add_task("Backend Development", "2024-02-10", 40, "backend")
    gantt.add_task("API Development", "2024-03-01", 25, "api")
    gantt.add_task("Integration Testing", "2024-03-20", 15, "testing")
    gantt.add_task("User Testing", "2024-04-01", 10, "testing")
    gantt.add_task("Bug Fixes", "2024-04-10", 12, "maintenance")
    gantt.add_task("Deployment", "2024-04-20", 5, "deployment")
    gantt.add_task("Documentation", "2024-04-15", 15, "documentation")
    
    gantt.create_gantt_chart(
        title="Software Development Project",
        save_path="software_project_gantt.png",
        time_unit="weeks"
    )
    
    return gantt

def create_research_project_example():
    """
    Create a research project example
    """
    gantt = GanttChartGenerator(figsize=(12, 8))
    
    # Add tasks for a research project
    gantt.add_task("Literature Review", "2024-01-01", 30, "research")
    gantt.add_task("Research Proposal", "2024-01-20", 15, "planning")
    gantt.add_task("Data Collection Setup", "2024-02-01", 20, "setup")
    gantt.add_task("Pilot Study", "2024-02-15", 25, "study")
    gantt.add_task("Main Data Collection", "2024-03-10", 45, "datacollection")
    gantt.add_task("Data Analysis", "2024-04-20", 30, "analysis")
    gantt.add_task("Results Interpretation", "2024-05-15", 20, "analysis")
    gantt.add_task("Paper Writing", "2024-06-01", 40, "writing")
    gantt.add_task("Peer Review", "2024-07-10", 15, "review")
    gantt.add_task("Revisions", "2024-07-25", 20, "writing")
    gantt.add_task("Final Submission", "2024-08-15", 5, "submission")
    
    gantt.create_gantt_chart(
        title="Research Project Timeline",
        save_path="research_project_gantt.png"
    )
    
    return gantt

def create_custom_project():
    """
    Create a custom project interactively
    """
    gantt = GanttChartGenerator()
    
    print("Creating a custom Gantt chart...")
    print("Enter task details (press Enter with empty name to finish):")
    
    while True:
        task_name = input("\nTask name: ").strip()
        if not task_name:
            break
        
        start_date = input("Start date (YYYY-MM-DD): ").strip()
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
        except ValueError:
            print("Invalid date format. Using default date.")
            start_date = "2024-01-01"
        
        try:
            duration = int(input("Duration (days): "))
        except ValueError:
            print("Invalid duration. Using 7 days.")
            duration = 7
        
        category = input("Category (optional): ").strip() or None
        
        gantt.add_task(task_name, start_date, duration, category)
    
    if gantt.tasks:
        title = input("\nChart title: ").strip() or "Custom Project"
        gantt.create_gantt_chart(title=title, save_path="custom_gantt.png")
    else:
        print("No tasks added.")
    
    return gantt

def demonstrate_csv_functionality():
    """
    Demonstrate CSV import/export functionality
    """
    print("\n=== CSV Functionality Demo ===")
    
    # Create a project and export to CSV
    gantt = GanttChartGenerator()
    gantt.create_sample_project()
    gantt.export_to_csv("sample_project.csv")
    
    # Create a new instance and load from CSV
    gantt2 = GanttChartGenerator()
    gantt2.load_from_csv("sample_project.csv")
    
    print("Successfully demonstrated CSV export and import!")

def create_iterative_project_example():
    """
    Create an iterative project example like the one in the second image
    """
    gantt = GanttChartGenerator(figsize=(16, 8))
    
    # Project based on the iterative development image
    # Tasks are added in logical order (top to bottom in Gantt chart)
    
    # Planning phase - starts first, goes at top
    gantt.add_task("Planificación y estudio de la implementación", "2023-01-02", 33, "planning")
    
    # Iterations in chronological order
    gantt.add_task("Iteración 1", "2023-01-09", 35, "iteration")
    gantt.add_task("Iteración 2", "2023-01-16", 56, "iteration") 
    gantt.add_task("Iteración 3", "2023-01-24", 49, "iteration")
    gantt.add_task("Iteración 4", "2023-01-31", 42, "iteration")
    gantt.add_task("Iteración 5", "2023-02-28", 28, "iteration")
    gantt.add_task("Iteración 6", "2023-04-04", 21, "iteration")
    gantt.add_task("Iteración 7", "2023-04-25", 91, "iteration")
    
    # Documentation runs parallel but is listed last in logical order
    gantt.add_task("Documentación", "2023-01-02", 224, "documentation")
    
    gantt.create_gantt_chart(
        title="Diagrama de Gantt: Planificación temporal del proyecto",
        save_path="iterative_project_gantt.png"
    )
    
    print(f"Created iterative project with {len(gantt.tasks)} tasks")
    print("Tasks are displayed in logical order (top to bottom)")
    
    return gantt

def main():
    """
    Main function to run examples
    """
    print("Gantt Chart Generator Examples")
    print("==============================")
    while True:
        print("\nChoose an example:")
        print("1. Original sample project (from first image)")
        print("2. Iterative project (from second image)")
        print("3. Software development project")
        print("4. Research project")
        print("5. Create custom project")
        print("6. CSV functionality demo")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-6): ").strip()
        
        if choice == "0":
            print("Goodbye!")
            break
        elif choice == "1":
            gantt = GanttChartGenerator(figsize=(14, 10))
            gantt.create_sample_project()
            gantt.create_gantt_chart(
                title="Diagrama de Gantt - Proyecto de Desarrollo",
                save_path="original_sample_gantt.png",
                time_unit="weeks"
            )
        elif choice == "2":
            create_iterative_project_example()
        elif choice == "3":
            create_software_project_example()
        elif choice == "4":
            create_research_project_example()
        elif choice == "5":
            create_custom_project()
        elif choice == "6":
            demonstrate_csv_functionality()
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
