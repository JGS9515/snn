"""
Clean test - minimal Gantt chart to verify equal heights
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime, timedelta

def create_clean_test():
    """Create a minimal test with 3 tasks to verify equal heights"""
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Task data
    tasks = [
        {"name": "Task 1 (TOP)", "start": 0, "duration": 10, "color": "#FFD700"},
        {"name": "Task 2 (MIDDLE)", "start": 5, "duration": 10, "color": "#87CEEB"}, 
        {"name": "Task 3 (BOTTOM)", "start": 10, "duration": 10, "color": "#FFA07A"}
    ]
    
    num_tasks = len(tasks)
    
    # Plot rectangles with EXACT same height
    for i, task in enumerate(tasks):
        y_pos = num_tasks - 1 - i  # First task at top
        
        print(f"Task {i+1}: y_pos = {y_pos}, height = 0.8")
        
        rect = patches.Rectangle(
            (task["start"], y_pos),  # x, y position
            task["duration"],        # width
            0.8,                    # height (EXACTLY the same for all)
            linewidth=1,
            edgecolor='black',
            facecolor=task["color"],
            alpha=0.8
        )
        ax.add_patch(rect)
    
    # Set up axes
    ax.set_xlim(0, 25)
    ax.set_ylim(-0.5, num_tasks - 0.5)
    
    # Labels (reversed to match rectangle positions)
    y_labels = [task["name"] for task in reversed(tasks)]
    ax.set_yticks(range(num_tasks))
    ax.set_yticklabels(y_labels, fontsize=10)
    
    # Set x-axis
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    ax.set_xticklabels(["Week 1", "Week 2", "Week 3", "Week 4", "Week 5", "Week 6"])
    
    ax.grid(True, alpha=0.3)
    ax.set_title("Height Test - All Bars Should Be EXACTLY Same Height", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("clean_height_test.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nCheck the generated chart:")
    print("- All 3 bars should have EXACTLY the same height")
    print("- Task 1 should be at the TOP (yellow)")
    print("- Task 2 should be in the MIDDLE (blue)")
    print("- Task 3 should be at the BOTTOM (salmon)")

if __name__ == "__main__":
    create_clean_test()
