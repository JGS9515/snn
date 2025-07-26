"""
Demonstration of proper Gantt chart task ordering
This script shows how tasks should be ordered in a Gantt chart - chronologically 
from top to bottom based on when they were added (logical project structure)
"""

from diagrama_grant import GanttChartGenerator
from datetime import datetime

def demonstrate_proper_ordering():
    """
    Demonstrate the difference between proper task ordering and date-sorted ordering
    """
    print("=== Gantt Chart Task Ordering Demo ===")
    print()
    
    # Create example based on the iterative project image
    gantt = GanttChartGenerator(figsize=(16, 10))
    
    # Add tasks in LOGICAL order (how they appear in the project structure)
    # This is the order they should appear in the Gantt chart from top to bottom
    
    print("Adding tasks in logical project order:")
    
    # 1. Planning comes first logically and chronologically
    gantt.add_task("Planificación y estudio de la implementación", "2023-01-02", 33, "planning")
    print("1. Planificación y estudio... (starts: 2023-01-02)")
    
    # 2-8. Iterations in sequence
    gantt.add_task("Iteración 1", "2023-01-09", 35, "iteration")
    print("2. Iteración 1 (starts: 2023-01-09)")
    
    gantt.add_task("Iteración 2", "2023-01-16", 56, "iteration") 
    print("3. Iteración 2 (starts: 2023-01-16)")
    
    gantt.add_task("Iteración 3", "2023-01-24", 49, "iteration")
    print("4. Iteración 3 (starts: 2023-01-24)")
    
    gantt.add_task("Iteración 4", "2023-01-31", 42, "iteration")
    print("5. Iteración 4 (starts: 2023-01-31)")
    
    gantt.add_task("Iteración 5", "2023-02-28", 28, "iteration")
    print("6. Iteración 5 (starts: 2023-02-28)")
    
    gantt.add_task("Iteración 6", "2023-04-04", 21, "iteration")
    print("7. Iteración 6 (starts: 2023-04-04)")
    
    gantt.add_task("Iteración 7", "2023-04-25", 91, "iteration")
    print("8. Iteración 7 (starts: 2023-04-25)")
    
    # 9. Documentation runs parallel but is listed last in project structure
    gantt.add_task("Documentación", "2023-01-02", 224, "documentation")
    print("9. Documentación (starts: 2023-01-02, runs parallel)")
    
    print()
    print("Tasks will appear in the Gantt chart from TOP to BOTTOM in the order above.")
    print("This matches the logical project structure, not chronological start dates.")
    print()
    
    # Generate the chart
    gantt.create_gantt_chart(
        title="Correct Gantt Chart: Tasks in Logical Order",
        save_path="correct_gantt_ordering.png"
    )
    
    print("Chart saved as: correct_gantt_ordering.png")
    print()
    print("Key points about proper Gantt chart ordering:")
    print("• First task added appears at the TOP")
    print("• Tasks flow from top to bottom in logical sequence")
    print("• Order reflects project structure, not just start dates")
    print("• Parallel tasks (like Documentation) are positioned based on importance")
    
    return gantt

def demonstrate_wrong_vs_right():
    """
    Show the difference between sorting by date vs. proper ordering
    """
    print("\n=== Comparison: Wrong vs Right Ordering ===")
    print()
    
    # Create a simple example to illustrate the difference
    gantt = GanttChartGenerator(figsize=(12, 6))
    
    # Add tasks in project logical order (not date order)
    gantt.add_task("Project Setup", "2024-01-15", 10, "setup")          # Starts 15th
    gantt.add_task("Requirements", "2024-01-01", 20, "requirements")    # Starts 1st (earlier!)
    gantt.add_task("Design", "2024-01-20", 15, "design")               # Starts 20th
    gantt.add_task("Implementation", "2024-01-10", 25, "implementation") # Starts 10th
    gantt.add_task("Testing", "2024-02-01", 10, "testing")             # Starts Feb 1st
    
    print("Tasks added in this logical order:")
    print("1. Project Setup (starts Jan 15)")
    print("2. Requirements (starts Jan 1) ← Earlier start date!")
    print("3. Design (starts Jan 20)")
    print("4. Implementation (starts Jan 10) ← Earlier start date!")
    print("5. Testing (starts Feb 1)")
    print()
    print("CORRECT: Chart shows tasks in the order above (1,2,3,4,5 from top)")
    print("WRONG: Would sort by start date (2,4,1,3,5 from top)")
    print()
    
    gantt.create_gantt_chart(
        title="Correct: Tasks in Logical Project Order",
        save_path="logical_order_demo.png"
    )
    
    return gantt

def main():
    """
    Main demonstration
    """
    print("Gantt Chart Task Ordering Demonstration")
    print("=====================================")
    print()
    print("This demo shows how to properly order tasks in a Gantt chart.")
    print("Tasks should appear from top to bottom in LOGICAL project order,")
    print("not necessarily in chronological start date order.")
    print()
    
    # Run demonstrations
    demonstrate_proper_ordering()
    demonstrate_wrong_vs_right()
    
    print("\nDemo complete! Check the generated PNG files to see the results.")

if __name__ == "__main__":
    main()
