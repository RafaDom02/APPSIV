# ============================================================================
# CÓDIGO DE VISUALIZACIONES INTERACTIVAS PARA NOTEBOOK MOT
# ============================================================================
# Copia este código al final de tu notebook APPSIV_mot_baseline_2025.ipynb

# CELDA 1: Instalación de librerías
"""
# Instalar librerías adicionales para visualización interactiva
!pip install plotly seaborn -q
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import warnings
warnings.filterwarnings('ignore')
"""

# CELDA 2: Extracción de métricas
"""
# Extraer métricas detalladas de los resultados
def extract_detailed_metrics():
    if not mot_accums:
        print("No hay datos de evaluación disponibles.")
        return None
    
    import motmetrics as mm
    mh = mm.metrics.create()
    
    # Obtener métricas para cada secuencia
    sequence_names = [str(s) for s in sequences if not s.no_gt]
    summary = mh.compute_many(
        mot_accums,
        metrics=mm.metrics.motchallenge_metrics,
        names=sequence_names,
        generate_overall=True
    )
    
    return summary, sequence_names

# Extraer información de tracking por secuencia
def extract_tracking_stats():
    tracking_stats = []
    
    for seq_name, results in results_seq.items():
        # Estadísticas básicas
        num_tracks = len(results)
        total_detections = sum(len(frames) for frames in results.values())
        
        # Duración promedio de tracks
        track_lengths = [len(frames) for frames in results.values()]
        avg_track_length = np.mean(track_lengths) if track_lengths else 0
        max_track_length = max(track_lengths) if track_lengths else 0
        min_track_length = min(track_lengths) if track_lengths else 0
        
        # Distribución de scores
        all_scores = []
        for track_frames in results.values():
            for frame_data in track_frames.values():
                if len(frame_data) > 4:  # Asegurar que hay score
                    all_scores.append(frame_data[4])
        
        avg_score = np.mean(all_scores) if all_scores else 0
        
        tracking_stats.append({
            'Secuencia': seq_name,
            'Num_Tracks': num_tracks,
            'Total_Detecciones': total_detections,
            'Longitud_Promedio': avg_track_length,
            'Longitud_Maxima': max_track_length,
            'Longitud_Minima': min_track_length,
            'Score_Promedio': avg_score
        })
    
    return pd.DataFrame(tracking_stats)

# Ejecutar extracción
metrics_data = extract_detailed_metrics()
tracking_df = extract_tracking_stats()

print("Datos extraídos exitosamente!")
print(f"Estadísticas de tracking para {len(tracking_df)} secuencias")
"""

# CELDA 3: Dashboard de métricas
"""
# Dashboard interactivo de métricas
def create_metrics_dashboard():
    if metrics_data is None:
        print("No hay métricas disponibles para visualizar.")
        return
    
    summary, sequence_names = metrics_data
    
    # Convertir a DataFrame para facilitar el manejo
    metrics_df = summary.reset_index()
    
    # Crear subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('MOTA por Secuencia', 'MOTP por Secuencia', 
                       'IDF1 vs MOTA', 'Distribución de Métricas Principales'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "box"}]]
    )
    
    # Filtrar solo las secuencias (no OVERALL)
    seq_data = metrics_df[metrics_df['index'].isin(sequence_names)]
    
    if not seq_data.empty:
        # 1. MOTA por secuencia
        fig.add_trace(
            go.Bar(
                x=seq_data['index'],
                y=seq_data['mota'] * 100,  # Convertir a porcentaje
                name='MOTA (%)',
                marker_color='lightblue',
                text=[f'{x:.1f}%' for x in seq_data['mota'] * 100],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # 2. MOTP por secuencia
        fig.add_trace(
            go.Bar(
                x=seq_data['index'],
                y=seq_data['motp'],
                name='MOTP',
                marker_color='lightcoral',
                text=[f'{x:.3f}' for x in seq_data['motp']],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        # 3. Scatter IDF1 vs MOTA
        fig.add_trace(
            go.Scatter(
                x=seq_data['mota'] * 100,
                y=seq_data['idf1'] * 100,
                mode='markers+text',
                text=seq_data['index'],
                textposition='top center',
                marker=dict(size=10, color='green'),
                name='Secuencias'
            ),
            row=2, col=1
        )
        
        # 4. Box plots de métricas principales
        metrics_to_plot = ['mota', 'motp', 'idf1', 'precision', 'recall']
        for metric in metrics_to_plot:
            if metric in seq_data.columns:
                fig.add_trace(
                    go.Box(
                        y=seq_data[metric] * (100 if metric != 'motp' else 1),
                        name=metric.upper(),
                        boxpoints='all'
                    ),
                    row=2, col=2
                )
    
    # Actualizar layout
    fig.update_layout(
        height=800,
        title_text="Dashboard de Métricas MOT",
        showlegend=True
    )
    
    # Actualizar ejes
    fig.update_xaxes(title_text="Secuencia", row=1, col=1)
    fig.update_yaxes(title_text="MOTA (%)", row=1, col=1)
    fig.update_xaxes(title_text="Secuencia", row=1, col=2)
    fig.update_yaxes(title_text="MOTP", row=1, col=2)
    fig.update_xaxes(title_text="MOTA (%)", row=2, col=1)
    fig.update_yaxes(title_text="IDF1 (%)", row=2, col=1)
    fig.update_yaxes(title_text="Valor de Métrica", row=2, col=2)
    
    fig.show()

create_metrics_dashboard()
"""

# CELDA 4: Análisis de comportamiento de tracking
"""
# Visualización del comportamiento de tracking
def create_tracking_analysis():
    # 1. Gráfico de estadísticas generales
    fig1 = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Número de Tracks por Secuencia', 'Distribución de Longitudes de Track',
                       'Scores Promedio por Secuencia', 'Total de Detecciones'),
        specs=[[{"secondary_y": False}, {"type": "histogram"}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Número de tracks por secuencia
    fig1.add_trace(
        go.Bar(
            x=tracking_df['Secuencia'],
            y=tracking_df['Num_Tracks'],
            name='Tracks',
            marker_color='skyblue',
            text=tracking_df['Num_Tracks'],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # Histograma de longitudes de track
    all_lengths = []
    for seq_name, results in results_seq.items():
        track_lengths = [len(frames) for frames in results.values()]
        all_lengths.extend(track_lengths)
    
    fig1.add_trace(
        go.Histogram(
            x=all_lengths,
            nbinsx=20,
            name='Distribución',
            marker_color='lightgreen'
        ),
        row=1, col=2
    )
    
    # Scores promedio
    fig1.add_trace(
        go.Scatter(
            x=tracking_df['Secuencia'],
            y=tracking_df['Score_Promedio'],
            mode='markers+lines',
            name='Score Promedio',
            marker=dict(size=10, color='red'),
            line=dict(color='red', width=2)
        ),
        row=2, col=1
    )
    
    # Total detecciones
    fig1.add_trace(
        go.Bar(
            x=tracking_df['Secuencia'],
            y=tracking_df['Total_Detecciones'],
            name='Detecciones',
            marker_color='orange',
            text=tracking_df['Total_Detecciones'],
            textposition='auto'
        ),
        row=2, col=2
    )
    
    fig1.update_layout(
        height=800,
        title_text="Análisis de Comportamiento de Tracking",
        showlegend=True
    )
    
    fig1.show()
    
    # 2. Análisis temporal de tracks
    create_temporal_analysis()

def create_temporal_analysis():
    \"\"\"Análisis temporal del comportamiento de tracks\"\"\"
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3
    
    for i, (seq_name, results) in enumerate(results_seq.items()):
        # Contar tracks activos por frame
        frame_counts = defaultdict(int)
        max_frame = 0
        
        for track_id, frames in results.items():
            for frame_idx in frames.keys():
                frame_counts[frame_idx] += 1
                max_frame = max(max_frame, frame_idx)
        
        # Crear serie temporal
        frames = list(range(max_frame + 1))
        counts = [frame_counts.get(f, 0) for f in frames]
        
        fig.add_trace(
            go.Scatter(
                x=frames,
                y=counts,
                mode='lines+markers',
                name=seq_name,
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=4)
            )
        )
    
    fig.update_layout(
        title="Evolución Temporal de Tracks Activos",
        xaxis_title="Frame",
        yaxis_title="Número de Tracks Activos",
        height=500,
        hovermode='x unified'
    )
    
    fig.show()

create_tracking_analysis()
"""

# CELDA 5: Análisis comparativo con estado del arte
"""
# Comparación con resultados de Tracktor++ (estado del arte)
def create_comparative_analysis():
    # Datos de referencia de Tracktor++ (del notebook)
    tracktor_results = {
        'MOT16-02': {'MOTA': 40.9, 'MOTP': 0.080, 'IDF1': 45.8, 'Precision': 99.8, 'Recall': 41.3},
        'MOT16-04': {'MOTA': 64.5, 'MOTP': 0.096, 'IDF1': 71.1, 'Precision': 99.8, 'Recall': 64.7},
        'MOT16-05': {'MOTA': 55.8, 'MOTP': 0.144, 'IDF1': 64.0, 'Precision': 98.1, 'Recall': 57.5},
        'MOT16-09': {'MOTA': 63.3, 'MOTP': 0.086, 'IDF1': 54.6, 'Precision': 99.1, 'Recall': 64.3},
        'MOT16-10': {'MOTA': 70.4, 'MOTP': 0.148, 'IDF1': 64.3, 'Precision': 98.0, 'Recall': 72.4},
        'MOT16-11': {'MOTA': 68.0, 'MOTP': 0.081, 'IDF1': 63.3, 'Precision': 98.9, 'Recall': 69.0},
        'MOT16-13': {'MOTA': 71.9, 'MOTP': 0.132, 'IDF1': 73.6, 'Precision': 97.6, 'Recall': 74.2}
    }
    
    if metrics_data is None:
        print("No hay métricas disponibles para comparación.")
        return
    
    summary, sequence_names = metrics_data
    our_results = summary.reset_index()
    
    # Preparar datos para comparación
    comparison_data = []
    
    for seq in sequence_names:
        if seq in our_results['index'].values:
            our_row = our_results[our_results['index'] == seq].iloc[0]
            
            # Nuestros resultados
            comparison_data.append({
                'Secuencia': seq,
                'Método': 'Nuestro Tracker',
                'MOTA': our_row.get('mota', 0) * 100,
                'MOTP': our_row.get('motp', 0),
                'IDF1': our_row.get('idf1', 0) * 100,
                'Precision': our_row.get('precision', 0) * 100,
                'Recall': our_row.get('recall', 0) * 100
            })
            
            # Resultados de Tracktor++ (si disponibles)
            if seq in tracktor_results:
                tr_data = tracktor_results[seq]
                comparison_data.append({
                    'Secuencia': seq,
                    'Método': 'Tracktor++',
                    'MOTA': tr_data['MOTA'],
                    'MOTP': tr_data['MOTP'],
                    'IDF1': tr_data['IDF1'],
                    'Precision': tr_data['Precision'],
                    'Recall': tr_data['Recall']
                })
    
    if not comparison_data:
        print("No hay datos suficientes para comparación.")
        return
    
    comp_df = pd.DataFrame(comparison_data)
    
    # Crear visualización comparativa
    metrics_to_compare = ['MOTA', 'IDF1', 'Precision', 'Recall']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f'Comparación {metric}' for metric in metrics_to_compare]
    )
    
    positions = [(1,1), (1,2), (2,1), (2,2)]
    colors = {'Nuestro Tracker': 'lightblue', 'Tracktor++': 'lightcoral'}
    
    for i, metric in enumerate(metrics_to_compare):
        row, col = positions[i]
        
        for method in comp_df['Método'].unique():
            method_data = comp_df[comp_df['Método'] == method]
            
            fig.add_trace(
                go.Bar(
                    x=method_data['Secuencia'],
                    y=method_data[metric],
                    name=f'{method} - {metric}',
                    marker_color=colors.get(method, 'gray'),
                    text=[f'{x:.1f}' for x in method_data[metric]],
                    textposition='auto',
                    legendgroup=method,
                    showlegend=(i == 0)  # Solo mostrar leyenda en el primer gráfico
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        height=800,
        title_text="Comparación con Estado del Arte (Tracktor++)",
        barmode='group'
    )
    
    fig.show()
    
    # Tabla de resumen
    print("\\n📊 Resumen de Comparación:")
    pivot_table = comp_df.pivot_table(index='Secuencia', columns='Método', values='MOTA', aggfunc='mean')
    if 'Tracktor++' in pivot_table.columns and 'Nuestro Tracker' in pivot_table.columns:
        pivot_table['Diferencia'] = pivot_table['Nuestro Tracker'] - pivot_table['Tracktor++']
        print(pivot_table.round(2))
    
    return comp_df

comparison_results = create_comparative_analysis()
"""

# CELDA 6: Explorador interactivo
"""
# Crear un explorador interactivo
def create_interactive_explorer():
    \"\"\"Crea un explorador interactivo de todos los resultados\"\"\"
    
    print("🚀 Explorador Interactivo de Resultados de Tracking")
    print("=" * 50)
    
    # Resumen general
    total_sequences = len(results_seq)
    total_tracks = sum(len(results) for results in results_seq.values())
    total_detections = sum(
        sum(len(frames) for frames in results.values()) 
        for results in results_seq.values()
    )
    
    print(f"📊 Resumen General:")
    print(f"   • Secuencias procesadas: {total_sequences}")
    print(f"   • Total de tracks: {total_tracks}")
    print(f"   • Total de detecciones: {total_detections}")
    
    if metrics_data:
        summary, sequence_names = metrics_data
        overall_row = summary.loc['OVERALL'] if 'OVERALL' in summary.index else None
        if overall_row is not None:
            print(f"   • MOTA promedio: {overall_row.get('mota', 0)*100:.1f}%")
            print(f"   • MOTP promedio: {overall_row.get('motp', 0):.3f}")
            print(f"   • IDF1 promedio: {overall_row.get('idf1', 0)*100:.1f}%")
    
    # Crear dashboard final
    fig = go.Figure()
    
    # Gráfico de radar para métricas generales
    if metrics_data:
        summary, sequence_names = metrics_data
        seq_data = summary.reset_index()
        seq_metrics = seq_data[seq_data['index'].isin(sequence_names)]
        
        if not seq_metrics.empty:
            # Calcular promedios
            avg_metrics = {
                'MOTA': seq_metrics['mota'].mean() * 100,
                'MOTP': (1 - seq_metrics['motp'].mean()) * 100,  # Invertir para que mayor sea mejor
                'IDF1': seq_metrics['idf1'].mean() * 100,
                'Precision': seq_metrics['precision'].mean() * 100,
                'Recall': seq_metrics['recall'].mean() * 100
            }
            
            fig.add_trace(go.Scatterpolar(
                r=list(avg_metrics.values()),
                theta=list(avg_metrics.keys()),
                fill='toself',
                name='Nuestro Tracker',
                line_color='blue'
            ))
            
            # Agregar referencia de Tracktor++ si está disponible
            tracktor_avg = {
                'MOTA': 61.7,
                'MOTP': (1 - 0.106) * 100,
                'IDF1': 65.0,
                'Precision': 99.1,
                'Recall': 62.6
            }
            
            fig.add_trace(go.Scatterpolar(
                r=list(tracktor_avg.values()),
                theta=list(tracktor_avg.keys()),
                fill='toself',
                name='Tracktor++ (Referencia)',
                line_color='red',
                opacity=0.6
            ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        title="Comparación de Métricas - Radar Chart",
        height=600
    )
    
    fig.show()
    
    # Recomendaciones
    print("\\n💡 Recomendaciones para Mejora:")
    if metrics_data:
        summary, sequence_names = metrics_data
        seq_data = summary.reset_index()
        seq_metrics = seq_data[seq_data['index'].isin(sequence_names)]
        
        if not seq_metrics.empty:
            avg_mota = seq_metrics['mota'].mean() * 100
            avg_precision = seq_metrics['precision'].mean() * 100
            avg_recall = seq_metrics['recall'].mean() * 100
            
            if avg_mota < 50:
                print("   🔴 MOTA bajo: Considerar mejorar el detector de objetos o el algoritmo de asociación")
            elif avg_mota < 65:
                print("   🟡 MOTA moderado: Hay margen de mejora en la asociación de datos")
            else:
                print("   🟢 MOTA bueno: El tracker está funcionando bien")
            
            if avg_precision < 90:
                print("   🔴 Precisión baja: Muchos falsos positivos, ajustar umbral de detección")
            
            if avg_recall < 60:
                print("   🔴 Recall bajo: Se están perdiendo muchas detecciones verdaderas")
    
    print("\\n📋 Secuencias disponibles para análisis detallado:")
    for i, seq_name in enumerate(results_seq.keys(), 1):
        num_tracks = len(results_seq[seq_name])
        print(f"   {i}. {seq_name} ({num_tracks} tracks)")
    
    return fig

explorer_fig = create_interactive_explorer()
"""

# CELDA 7: Análisis detallado por secuencia
"""
# Análisis detallado de una secuencia específica
def analyze_sequence_detail(sequence_name=None):
    if not results_seq:
        print("No hay resultados de tracking disponibles.")
        return
    
    # Si no se especifica secuencia, usar la primera disponible
    if sequence_name is None:
        sequence_name = list(results_seq.keys())[0]
    
    if sequence_name not in results_seq:
        print(f"Secuencia {sequence_name} no encontrada.")
        return
    
    results = results_seq[sequence_name]
    
    print(f"\\n🔍 Análisis detallado de la secuencia: {sequence_name}")
    
    # Estadísticas básicas
    num_tracks = len(results)
    track_lengths = [len(frames) for frames in results.values()]
    total_detections = sum(track_lengths)
    
    print(f"📈 Estadísticas básicas:")
    print(f"   • Número total de tracks: {num_tracks}")
    print(f"   • Total de detecciones: {total_detections}")
    print(f"   • Longitud promedio de track: {np.mean(track_lengths):.1f} frames")
    print(f"   • Track más largo: {max(track_lengths)} frames")
    print(f"   • Track más corto: {min(track_lengths)} frames")
    
    # Crear visualizaciones
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'Duración de Tracks - {sequence_name}',
            f'Distribución de Scores - {sequence_name}',
            f'Actividad por Frame - {sequence_name}',
            f'Mapa de Calor de Tracks - {sequence_name}'
        ),
        specs=[[{"secondary_y": False}, {"type": "histogram"}],
               [{"secondary_y": False}, {"type": "heatmap"}]]
    )
    
    # 1. Duración de tracks
    track_ids = list(results.keys())
    fig.add_trace(
        go.Bar(
            x=[f'Track {tid}' for tid in track_ids[:20]],  # Limitar a 20 para legibilidad
            y=track_lengths[:20],
            name='Duración',
            marker_color='lightblue'
        ),
        row=1, col=1
    )
    
    # 2. Distribución de scores
    all_scores = []
    for track_frames in results.values():
        for frame_data in track_frames.values():
            if len(frame_data) > 4:
                all_scores.append(frame_data[4])
    
    if all_scores:
        fig.add_trace(
            go.Histogram(
                x=all_scores,
                nbinsx=30,
                name='Scores',
                marker_color='lightgreen'
            ),
            row=1, col=2
        )
    
    # 3. Actividad por frame
    frame_activity = defaultdict(int)
    max_frame = 0
    
    for track_frames in results.values():
        for frame_idx in track_frames.keys():
            frame_activity[frame_idx] += 1
            max_frame = max(max_frame, frame_idx)
    
    frames = list(range(max_frame + 1))
    activity = [frame_activity.get(f, 0) for f in frames]
    
    fig.add_trace(
        go.Scatter(
            x=frames,
            y=activity,
            mode='lines+markers',
            name='Tracks Activos',
            line=dict(color='red', width=2),
            marker=dict(size=4)
        ),
        row=2, col=1
    )
    
    # 4. Mapa de calor de tracks (presencia por frame)
    if num_tracks > 0 and max_frame > 0:
        # Crear matriz de presencia
        presence_matrix = np.zeros((min(num_tracks, 50), min(max_frame + 1, 200)))  # Limitar tamaño
        
        for i, (track_id, track_frames) in enumerate(list(results.items())[:50]):
            for frame_idx in track_frames.keys():
                if frame_idx < presence_matrix.shape[1]:
                    presence_matrix[i, frame_idx] = 1
        
        fig.add_trace(
            go.Heatmap(
                z=presence_matrix,
                colorscale='Blues',
                showscale=False,
                name='Presencia'
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        height=800,
        title_text=f"Análisis Detallado - {sequence_name}",
        showlegend=True
    )
    
    fig.show()
    
    return {
        'sequence_name': sequence_name,
        'num_tracks': num_tracks,
        'total_detections': total_detections,
        'track_lengths': track_lengths,
        'scores': all_scores
    }

# Analizar la primera secuencia disponible
if results_seq:
    first_sequence = list(results_seq.keys())[0]
    sequence_analysis = analyze_sequence_detail(first_sequence)
else:
    print("No hay secuencias disponibles para analizar.")
"""

# CELDA 8: Exportar resultados
"""
# Función para exportar todos los análisis
def export_analysis_results():
    \"\"\"Exporta todos los resultados del análisis a archivos\"\"\"
    
    print("💾 Exportando resultados del análisis...")
    
    # 1. Exportar estadísticas de tracking
    if not tracking_df.empty:
        tracking_df.to_csv('tracking_statistics.csv', index=False)
        print("   ✅ Estadísticas de tracking guardadas en 'tracking_statistics.csv'")
    
    # 2. Exportar métricas de evaluación
    if metrics_data:
        summary, sequence_names = metrics_data
        summary.to_csv('evaluation_metrics.csv')
        print("   ✅ Métricas de evaluación guardadas en 'evaluation_metrics.csv'")
    
    # 3. Exportar comparación con estado del arte
    if 'comparison_results' in globals() and comparison_results is not None:
        comparison_results.to_csv('comparison_with_sota.csv', index=False)
        print("   ✅ Comparación con estado del arte guardada en 'comparison_with_sota.csv'")
    
    # 4. Crear reporte HTML
    html_report = f\"\"\"
    <!DOCTYPE html>
    <html>
    <head>
        <title>Reporte de Análisis MOT</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 10px; }}
            .metric {{ background-color: #e8f4fd; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .warning {{ background-color: #fff3cd; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .success {{ background-color: #d4edda; padding: 10px; margin: 10px 0; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎯 Reporte de Análisis - Multi-Object Tracking</h1>
            <p>Generado automáticamente desde el notebook de APPSIV</p>
        </div>
        
        <h2>📊 Resumen Ejecutivo</h2>
        <div class="metric">
            <strong>Secuencias procesadas:</strong> {len(results_seq)}<br>
            <strong>Total de tracks:</strong> {sum(len(results) for results in results_seq.values())}<br>
            <strong>Total de detecciones:</strong> {sum(sum(len(frames) for frames in results.values()) for results in results_seq.values())}
        </div>
    \"\"\"
    
    if metrics_data:
        summary, sequence_names = metrics_data
        overall_row = summary.loc['OVERALL'] if 'OVERALL' in summary.index else None
        if overall_row is not None:
            html_report += f\"\"\"
            <h2>🎯 Métricas Principales</h2>
            <div class="metric">
                <strong>MOTA:</strong> {overall_row.get('mota', 0)*100:.1f}%<br>
                <strong>MOTP:</strong> {overall_row.get('motp', 0):.3f}<br>
                <strong>IDF1:</strong> {overall_row.get('idf1', 0)*100:.1f}%<br>
                <strong>Precision:</strong> {overall_row.get('precision', 0)*100:.1f}%<br>
                <strong>Recall:</strong> {overall_row.get('recall', 0)*100:.1f}%
            </div>
            \"\"\"
    
    html_report += \"\"\"
        <h2>📈 Archivos Generados</h2>
        <ul>
            <li>tracking_statistics.csv - Estadísticas detalladas por secuencia</li>
            <li>evaluation_metrics.csv - Métricas de evaluación MOT</li>
            <li>comparison_with_sota.csv - Comparación con estado del arte</li>
            <li>analysis_report.html - Este reporte</li>
        </ul>
        
        <h2>💡 Próximos Pasos</h2>
        <div class="warning">
            <p>Para mejorar el rendimiento del tracker, considera:</p>
            <ul>
                <li>Ajustar parámetros del detector de objetos</li>
                <li>Optimizar el algoritmo de asociación de datos</li>
                <li>Implementar filtros de Kalman para predicción de movimiento</li>
                <li>Ajustar umbrales de IoU y confianza</li>
            </ul>
        </div>
    </body>
    </html>
    \"\"\"
    
    with open('analysis_report.html', 'w', encoding='utf-8') as f:
        f.write(html_report)
    
    print("   ✅ Reporte HTML guardado en 'analysis_report.html'")
    print("\\n🎉 ¡Exportación completada! Todos los archivos están listos.")
    
    # Mostrar el reporte HTML en el notebook
    display(HTML(html_report))

# Ejecutar exportación
export_analysis_results()
"""

print("📋 INSTRUCCIONES DE USO:")
print("=" * 50)
print("1. Copia cada bloque de código (entre comillas triples) a una nueva celda en tu notebook")
print("2. Ejecuta las celdas en orden secuencial")
print("3. Las visualizaciones aparecerán automáticamente")
print("4. Los archivos se exportarán al directorio actual")
print("5. Interactúa con los gráficos usando zoom, hover y filtros")
print("\n🎯 ¡Disfruta explorando tus resultados de tracking!") 