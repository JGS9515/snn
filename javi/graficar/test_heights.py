"""
Test script to verify the height fix for Gantt chart tasks
"""

from diagrama_grant import GanttChartGenerator

def test_task_heights():
    """Test that all tasks have the same height"""
    print("Testing task heights fix...")
    
    gantt = GanttChartGenerator(figsize=(10, 6))
    
    # Add 5 simple tasks
    gantt.add_task("Task 1", "2024-01-01", 7, "test")
    gantt.add_task("Task 2", "2024-01-08", 7, "test") 
    gantt.add_task("Task 3", "2024-01-15", 7, "test")
    gantt.add_task("Task 4", "2024-01-22", 7, "test")
    gantt.add_task("Task 5", "2024-01-29", 7, "test")
    
    print("Tasks added in order:")
    for i, task in enumerate(gantt.tasks, 1):
        print(f"{i}. {task['task_name']} (should appear at position {i} from top)")
    
    gantt.create_gantt_chart(
        title="Height Test - All Tasks Should Have Same Height",
        save_path="height_test.png",
        time_unit="weeks"
    )
    
    print("\nChart generated. Check that:")
    print("1. All task bars have the same height")
    print("2. Task 1 appears at the TOP")
    print("3. Tasks flow downward in order (1,2,3,4,5)")
    print("4. No task appears shorter than others")

if __name__ == "__main__":
    test_task_heights()
