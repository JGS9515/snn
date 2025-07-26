"""
Replica exacta del diagrama de Gantt de la imagen
Este script recrea el diagrama mostrado en la imagen adjunta
"""

from diagrama_grant import GanttChartGenerator
import matplotlib.pyplot as plt

def create_exact_replica():
    """
    Crea una réplica exacta del diagrama de la imagen
    """
    gantt = GanttChartGenerator(figsize=(16, 12))
    
    # Fase de investigación y búsqueda de información
    gantt.add_task("Búsqueda de información sobre tecnología necesaria", "2024-01-01", 45, "research")
    
    # Tecnología de sensores
    gantt.add_task("Tecnología de sensores", "2024-01-15", 50, "sensors")
    gantt.add_task("    Aplicación web", "2024-02-01", 25, "web")
    gantt.add_task("    Servicios REST", "2024-02-10", 20, "api")
    gantt.add_task("    Aplicación Android", "2024-02-20", 30, "mobile")
    gantt.add_task("    Búsqueda de bibliografía", "2024-03-01", 20, "research")
    gantt.add_task("    Búsqueda de bibliografía (cont.)", "2024-03-15", 15, "research")
    
    # Desarrollo de sensores (hardware)
    gantt.add_task("Desarrollo de los sensores (hardware)", "2024-04-01", 35, "hardware")
    gantt.add_task("Desarrollo de los sensores (hardware) - Fase 2", "2024-04-20", 40, "hardware")
    
    # Desarrollo software
    gantt.add_task("Desarrollo software", "2024-05-01", 80, "software")
    gantt.add_task("    Sensores", "2024-05-10", 35, "sensors")
    gantt.add_task("    Diseño", "2024-05-25", 25, "design")
    gantt.add_task("    Implementación", "2024-06-10", 40, "implementation")
    
    # Aplicación web
    gantt.add_task("Aplicación web", "2024-07-01", 50, "web")
    gantt.add_task("    Diseño", "2024-07-10", 20, "design")
    gantt.add_task("    Implementación", "2024-07-25", 35, "implementation")
    gantt.add_task("    Implementación de servicios REST", "2024-08-10", 25, "api")
    
    # Aplicación Android
    gantt.add_task("Aplicación Android", "2024-09-01", 45, "mobile")
    gantt.add_task("    Diseño", "2024-09-10", 15, "design")
    gantt.add_task("    Implementación", "2024-09-20", 30, "implementation")
    
    # Pruebas y finalización
    gantt.add_task("Pruebas software", "2024-10-15", 20, "testing")
    gantt.add_task("Corrección de posibles errores", "2024-11-01", 15, "testing")
    gantt.add_task("Instalación de software", "2024-11-10", 8, "deployment")
    gantt.add_task("Preparación del servidor", "2024-11-15", 10, "deployment")
    gantt.add_task("Despliegue de aplicaciones", "2024-11-20", 12, "deployment")
    
    # Documentación
    gantt.add_task("Realización de la documentación", "2024-12-01", 35, "documentation")
    gantt.add_task("    Memoria", "2024-12-05", 25, "documentation")
    gantt.add_task("    Manuales", "2024-12-20", 15, "documentation")
    
    return gantt

def customize_colors():
    """
    Personaliza los colores para que coincidan mejor con la imagen
    """
    # Colores similares a los de la imagen original
    custom_colors = {
        'research': '#FFD700',      # Dorado para búsqueda/investigación
        'sensors': '#87CEEB',       # Azul cielo para sensores
        'web': '#FFA07A',          # Salmón claro para web
        'api': '#98FB98',          # Verde claro para API/servicios
        'mobile': '#DDA0DD',       # Ciruela para móvil
        'hardware': '#F0E68C',     # Caqui para hardware
        'software': '#FFB6C1',     # Rosa claro para software
        'design': '#B0E0E6',       # Azul polvo para diseño
        'implementation': '#FFDAB9', # Durazno para implementación
        'testing': '#E6E6FA',      # Lavanda para pruebas
        'deployment': '#F5DEB3',   # Trigo para despliegue
        'documentation': '#D3D3D3'  # Gris claro para documentación
    }
    return custom_colors

def main():
    """
    Función principal para crear el diagrama
    """
    print("Creando diagrama de Gantt - Réplica exacta")
    print("==========================================")
    
    # Crear el diagrama
    gantt = create_exact_replica()
    
    # Mostrar información del proyecto
    print(f"Proyecto creado con {len(gantt.tasks)} tareas")
    print("\nTareas por categoría:")
    
    categories = {}
    for task in gantt.tasks:
        cat = task['category'] or 'Sin categoría'
        categories[cat] = categories.get(cat, 0) + 1
    
    for cat, count in categories.items():
        print(f"  {cat}: {count} tareas")
    
    # Generar el diagrama
    try:
        gantt.create_gantt_chart(
            title="Diagrama de Gantt - Proyecto de Desarrollo de Tecnología IoT",
            save_path="diagrama_gantt_replica.png"
        )
        print("\n✅ Diagrama creado exitosamente!")
        print("📁 Archivo guardado como: diagrama_gantt_replica.png")
        
        # Exportar datos
        gantt.export_to_csv("proyecto_replica.csv")
        print("📊 Datos exportados a: proyecto_replica.csv")
        
    except Exception as e:
        print(f"❌ Error al crear el diagrama: {e}")
        print("💡 Intenta ejecutar el script en un entorno con interfaz gráfica")

if __name__ == "__main__":
    main()
