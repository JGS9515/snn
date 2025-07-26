"""
Test script to verify that tasks are displayed in the correct order
First task at the top, subsequent tasks filling from top to bottom
"""

from diagrama_grant import GanttChartGenerator

def test_task_order():
    """
    Test that tasks appear in the order they were added
    """
    print("Testing task order (first task at top, filling top to bottom)...")
    
    gantt = GanttChartGenerator(figsize=(12, 8))
    
    # Add tasks in a specific order with different start dates
    # to verify they appear in the order added, not sorted by date
    gantt.add_task("1. First Task (starts March)", "2024-03-01", 10, "phase1")
    gantt.add_task("2. Second Task (starts January)", "2024-01-01", 15, "phase2")
    gantt.add_task("3. Third Task (starts February)", "2024-02-01", 20, "phase3")
    gantt.add_task("4. Fourth Task (starts April)", "2024-04-01", 12, "phase4")
    gantt.add_task("5. Fifth Task (starts January)", "2024-01-15", 8, "phase1")
    
    print("Tasks added in this order:")
    for i, task in enumerate(gantt.tasks, 1):
        print(f"  {i}. {task['task_name']} (starts: {task['start_date'].strftime('%Y-%m-%d')})")
    
    print("\nGenerating chart...")
    print("The chart should show tasks in the same order as above,")
    print("with 'First Task' at the top and 'Fifth Task' at the bottom.")
    
    try:
        gantt.create_gantt_chart(
            title="Test: Task Order (Top to Bottom)",
            save_path="test_task_order.png"
        )
        print("✅ Chart created successfully!")
        print("📁 Check 'test_task_order.png' to verify the order")
    except Exception as e:
        print(f"❌ Error creating chart: {e}")

if __name__ == "__main__":
    test_task_order()
