"""
Diagnostic test for Gantt chart task positioning
This will help identify why the first task appears smaller
"""

from diagrama_grant import GanttChartGenerator

def test_equal_heights():
    """Test that all tasks have exactly the same height"""
    print("=== DIAGNOSTIC TEST FOR EQUAL HEIGHTS ===")
    
    gantt = GanttChartGenerator(figsize=(12, 6))
    
    # Add simple tasks with same duration
    gantt.add_task("First Task (should be at TOP)", "2024-01-01", 14, "test1")
    gantt.add_task("Second Task", "2024-01-15", 14, "test2") 
    gantt.add_task("Third Task", "2024-01-29", 14, "test3")
    
    print("Tasks added:")
    for i, task in enumerate(gantt.tasks):
        print(f"{i+1}. {task['task_name']} - Duration: {task['duration']} days")
    
    print("\nExpected order in chart (top to bottom):")
    print("1. First Task (should be at TOP) - YELLOW")
    print("2. Second Task - BLUE") 
    print("3. Third Task - SALMON")
    print("\nAll bars should have EXACTLY the same height!")
    
    # Generate test chart
    gantt.create_gantt_chart(
        title="Height Test - All Bars Should Be Same Height",
        save_path="height_diagnostic.png",
        time_unit="weeks"
    )
    
    print("\nIf the first task (yellow) still appears smaller:")
    print("- Check the matplotlib window carefully")
    print("- The issue might be in the y-axis positioning")
    print("- All rectangles should have height = 0.8")

def test_positioning_values():
    """Print the exact positioning values"""
    print("\n=== POSITIONING VALUES TEST ===")
    
    gantt = GanttChartGenerator()
    gantt.add_task("Task A", "2024-01-01", 7, "test")
    gantt.add_task("Task B", "2024-01-08", 7, "test")
    gantt.add_task("Task C", "2024-01-15", 7, "test")
    
    ordered_tasks = gantt.tasks.copy()
    print(f"Number of tasks: {len(ordered_tasks)}")
    
    for i, task in enumerate(ordered_tasks):
        y_pos = len(ordered_tasks) - 1 - i
        print(f"Task {i+1}: '{task['task_name']}' -> y_pos = {y_pos}")
    
    print("\nExpected y positions (bottom to top):")
    print("y=0: Task C (bottom)")
    print("y=1: Task B (middle)")  
    print("y=2: Task A (top)")

if __name__ == "__main__":
    test_equal_heights()
    test_positioning_values()
