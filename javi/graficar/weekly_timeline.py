"""
Weekly Timeline Example - Matching the image format
This script creates a Gantt chart that shows project timeline in weeks
"""

from diagrama_grant import GanttChartGenerator
from datetime import datetime

def create_weekly_project_example():
    """
    Create a project example that clearly shows weekly timeline
    Similar to the iterative project in your image
    """
    gantt = GanttChartGenerator(figsize=(16, 8))
    
    # Create project with tasks that align to weeks
    # Using Monday 2023-01-02 as start date (Week 1)
    
    print("Creating project with weekly timeline...")
    print("=" * 50)
    
    # Planning phase - Week 1-5 (5 weeks)
    gantt.add_task("Planificación y estudio de la implementación", "2023-01-02", 33, "planning")
    
    # Iteration 1 - Week 2-7 (5 weeks) 
    gantt.add_task("Iteración 1", "2023-01-09", 35, "iteration")
    
    # Iteration 2 - Week 3-11 (8 weeks)
    gantt.add_task("Iteración 2", "2023-01-16", 56, "iteration")
    
    # Iteration 3 - Week 4-11 (7 weeks)
    gantt.add_task("Iteración 3", "2023-01-23", 49, "iteration")
    
    # Iteration 4 - Week 5-11 (6 weeks)
    gantt.add_task("Iteración 4", "2023-01-30", 42, "iteration")
    
    # Iteration 5 - Week 9-13 (4 weeks)
    gantt.add_task("Iteración 5", "2023-02-27", 28, "iteration")
    
    # Iteration 6 - Week 14-17 (3 weeks)
    gantt.add_task("Iteración 6", "2023-04-03", 21, "iteration")
    
    # Iteration 7 - Week 18-31 (13 weeks)
    gantt.add_task("Iteración 7", "2023-04-24", 91, "iteration")
    
    # Documentation - Week 1-33 (runs parallel, 32 weeks)
    gantt.add_task("Documentación", "2023-01-02", 224, "documentation")
    
    # Print detailed timeline
    gantt.print_project_summary()
    
    # Generate chart
    gantt.create_gantt_chart(
        title="Diagrama de Gantt: Planificación temporal del proyecto (Semanas)",
        save_path="gantt_weekly_timeline.png"
    )
    
    return gantt

def create_simple_weekly_example():
    """
    Create a simple example to demonstrate weekly planning
    """
    gantt = GanttChartGenerator(figsize=(12, 6))
    
    print("\nSimple Weekly Example:")
    print("=" * 30)
    
    # Start on Monday, January 2, 2024
    start_date = "2024-01-01"  # Monday
    
    # Week 1-2: Planning (2 weeks = 14 days)
    gantt.add_task("Project Planning", start_date, 14, "planning")
    
    # Week 3: Requirements (1 week = 7 days)
    gantt.add_task("Requirements Analysis", "2024-01-15", 7, "analysis")
    
    # Week 4-6: Design (3 weeks = 21 days)
    gantt.add_task("System Design", "2024-01-22", 21, "design")
    
    # Week 7-10: Development (4 weeks = 28 days)
    gantt.add_task("Development", "2024-02-12", 28, "development")
    
    # Week 11-12: Testing (2 weeks = 14 days)
    gantt.add_task("Testing", "2024-03-11", 14, "testing")
    
    # Week 13: Deployment (1 week = 7 days)
    gantt.add_task("Deployment", "2024-03-25", 7, "deployment")
    
    gantt.print_project_summary()
    
    gantt.create_gantt_chart(
        title="Simple Project: Weekly Timeline (13 weeks total)",
        save_path="simple_weekly_gantt.png"
    )
    
    return gantt

def main():
    """
    Main function to demonstrate weekly timeline
    """
    print("Gantt Chart - Weekly Timeline Examples")
    print("=" * 40)
    
    choice = input("\nChoose example:\n1. Iterative project (matches your image)\n2. Simple weekly example\n3. Both\nChoice (1-3): ")
    
    if choice == "1":
        create_weekly_project_example()
    elif choice == "2":
        create_simple_weekly_example()
    elif choice == "3":
        create_weekly_project_example()
        create_simple_weekly_example()
    else:
        print("Invalid choice, showing both examples:")
        create_weekly_project_example()
        create_simple_weekly_example()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("• Timeline is now shown in WEEKS (Semana 1, Semana 2, etc.)")
    print("• Each vertical line represents 1 week (7 days)")
    print("• Project summary shows duration in weeks")
    print("• Tasks are aligned to weekly boundaries for easier reading")
    print("=" * 60)

if __name__ == "__main__":
    main()
