import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime
from plotly.subplots import make_subplots
from tabs.utils import *

# ================================================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ================================================================================================

import os

# Ruta absoluta al favicon
FAVICON_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../recursos/favicon.png'))

st.set_page_config(
    page_title="Trading Analytics App",
    page_icon=FAVICON_PATH,
    layout="wide"
)

# ================================================================================================
# INICIALIZACIÓN DEL ESTADO
# ================================================================================================

def initialize_session_state():
    """Inicializa el estado de la sesión"""
    # Estados de los pasos del pipeline
    if 'step_load' not in st.session_state:
        st.session_state.step_load = False
    if 'step_target' not in st.session_state:
        st.session_state.step_target = False
    if 'step_split' not in st.session_state:
        st.session_state.step_split = False
    if 'step_feature_selection' not in st.session_state:
        st.session_state.step_feature_selection = False
    if 'step_extraction' not in st.session_state:
        st.session_state.step_extraction = False
    if 'step_validation' not in st.session_state:
        st.session_state.step_validation = False
    if 'step_ensemble' not in st.session_state:
        st.session_state.step_ensemble = False
    if 'step_optimization' not in st.session_state:  
        st.session_state.step_optimization = False  
    if 'step_sizing' not in st.session_state:
        st.session_state.step_sizing = False 
    
    # DataFrames del pipeline
    if 'df_raw' not in st.session_state:
        st.session_state.df_raw = None
    if 'df_with_target' not in st.session_state:
        st.session_state.df_with_target = None
    if 'train_df' not in st.session_state:
        st.session_state.train_df = None
    if 'test_df' not in st.session_state:
        st.session_state.test_df = None
    
    # Identificadores para evitar re-procesamiento
    if 'current_file_id' not in st.session_state:
        st.session_state.current_file_id = None

# ================================================================================================
# FUNCIONES AUXILIARES PARA GRÁFICOS
# ================================================================================================

def create_price_chart(df, title="Evolución del Precio de Cierre"):
    """Crea un gráfico de línea del precio de cierre"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#1f77b4', width=1.5),
        hovertemplate='<b>Fecha:</b> %{x}<br><b>Close:</b> %{y:.5f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20)),
        xaxis_title='Fecha',
        yaxis_title='Precio',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=False
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

def create_split_chart(df, train_df, test_df):
    """Crea gráfico del close con colores diferentes para train y test"""
    fig = go.Figure()
    
    # Gráfico del conjunto de entrenamiento
    if len(train_df) > 0:
        fig.add_trace(go.Scatter(
            x=train_df.index,
            y=train_df['Close'],
            mode='lines',
            name='Train',
            line=dict(color='#2E8B57', width=1.5),
            hovertemplate='<b>Fecha:</b> %{x}<br><b>Close:</b> %{y:.5f}<br><b>Set:</b> Train<extra></extra>'
        ))
    
    # Gráfico del conjunto de prueba
    if len(test_df) > 0:
        fig.add_trace(go.Scatter(
            x=test_df.index,
            y=test_df['Close'],
            mode='lines',
            name='Test',
            line=dict(color='#DC143C', width=1.5),
            hovertemplate='<b>Fecha:</b> %{x}<br><b>Close:</b> %{y:.5f}<br><b>Set:</b> Test<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text='División Train/Test del Precio de Cierre', x=0.5, font=dict(size=20)),
        xaxis_title='Fecha',
        yaxis_title='Precio',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

# ================================================================================================
# FUNCIÓN PARA ANÁLISIS DE VALORES NO NUMÉRICOS
# ================================================================================================

def analyze_non_numeric_values(df):
    """
    Analiza valores no numéricos en el DataFrame
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame a analizar
        
    Retorna:
    --------
    dict : Diccionario con información sobre valores problemáticos
    """
    results = {
        'columns_with_issues': {},
        'total_columns': len(df.columns),
        'problematic_columns': 0
    }
    
    # Columnas a ignorar (no son indicadores técnicos)
    ignore_cols = ['Date', 'Open', 'High', 'Low', 'Close']
    
    for col in df.columns:
        if col in ignore_cols:
            continue
            
        col_info = {
            'total_rows': len(df),
            'non_null_rows': df[col].notna().sum(),
            'null_rows': df[col].isna().sum(),
            'inf_values': 0,
            'neg_inf_values': 0,
            'non_numeric_values': 0,
            'problematic_samples': []
        }
        
        # Verificar si la columna es numérica
        try:
            # Convertir a numérico para detectar problemas
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            
            # Contar infinitos
            if np.any(np.isinf(numeric_series.dropna())):
                col_info['inf_values'] = np.sum(np.isposinf(numeric_series))
                col_info['neg_inf_values'] = np.sum(np.isneginf(numeric_series))
            
            # Contar valores que no se pudieron convertir a numérico
            non_numeric_mask = df[col].notna() & numeric_series.isna()
            col_info['non_numeric_values'] = non_numeric_mask.sum()
            
            # Obtener muestras de valores problemáticos
            if col_info['non_numeric_values'] > 0:
                problematic_values = df.loc[non_numeric_mask, col].unique()[:5]  # Máximo 5 ejemplos
                col_info['problematic_samples'] = list(problematic_values)
        
        except Exception as e:
            col_info['error'] = str(e)
        
        # Si hay problemas, agregar a los resultados
        total_issues = (col_info['inf_values'] + col_info['neg_inf_values'] + 
                       col_info['non_numeric_values'])
        
        if total_issues > 0:
            results['columns_with_issues'][col] = col_info
            results['problematic_columns'] += 1
    
    return results

# ================================================================================================
# TABS DEL PIPELINE
# ================================================================================================


def optimization_tab():
    """Tab OPTIMIZATION - Análisis de optimización MT5"""
    st.header("⚙️ OPTIMIZACIÓN - Análisis de Optimización MT5")
    st.markdown("---")
    
    # Información del proceso
    st.info(
        " **Proceso**: Sube el reporte XML de optimización de MT5 → "
        "Análisis estadístico → Rangos recomendados para parámetros"
    )
    
 
    
    # ================================================================================================
    # UPLOADER DE ARCHIVO XML
    # ================================================================================================
    
    st.markdown("### 📁 Cargar Reporte de Optimización")
    
    uploaded_xml = st.file_uploader(
        "Arrastra y suelta tu archivo XML de optimización MT5",
        type=['xml'],
        help="Archivo generado por MetaTrader 5 después de ejecutar una optimización"
    )
    
    if uploaded_xml is None:
        st.info("Carga un archivo XML para comenzar el análisis")
        
        # Mostrar instrucciones
        with st.expander("¿Cómo generar el archivo XML en MT5?", expanded=False):
            st.markdown("""
            **Pasos para obtener el reporte de optimización:**
            
            1. **Optimizar EA en MT5**: Strategy Tester → Optimization
            2. **Finalizar optimización**: Esperar a que termine el proceso
            3. **Exportar resultados**: Click derecho en tabla de resultados → "Save as Report"
            4. **Seleccionar formato**: Elegir "XML files (*.xml)"
            5. **Guardar archivo**: Dar nombre y guardar
            6. **Subir aquí**: Usar el uploader de arriba
            
            **Requisitos del archivo:**
            - Debe contener columna "Result" con los valores de fitness
            - Debe incluir columnas de parámetros optimizados
            - Formato XML estándar de MT5
            - Los valores deben ser numericos, no se podrán optimizar parametros booleanos o de texto
            """)
        return
    
    # ================================================================================================
    # PROCESAR ARCHIVO XML
    # ================================================================================================
    
    if uploaded_xml is not None:
        
        with st.spinner('📊 Analizando archivo XML...'):
            
            # Analizar archivo
            analisis_xml = analizar_archivo_optimizacion_mt5(uploaded_xml)
            
            if not analisis_xml['success']:
                st.error(f"❌ Error procesando archivo: {analisis_xml['error']}")
                return
            
            # Obtener DataFrame
            df_optimizacion = analisis_xml['dataframe']
            
            st.success(f"✅ Archivo procesado: {analisis_xml['total_filas']:,} filas, {len(analisis_xml['columnas'])} columnas")
            
            # Mostrar información básica
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📊 Total Pruebas", f"{len(df_optimizacion):,}")
            
            with col2:
                if 'Result' in df_optimizacion.columns:
                    resultados_positivos = len(df_optimizacion[df_optimizacion['Result'] > 0])
                    st.metric("✅ Resultados > 0", f"{resultados_positivos:,}")
                else:
                    st.metric("❌ Sin columna 'Result'", "0")
            
            with col3:
                st.metric("📋 Columnas Detectadas", len(analisis_xml['columnas']))
            
            # Mostrar muestra de datos
            st.markdown("### Vista Previa de Datos")
            st.dataframe(df_optimizacion.head(10), use_container_width=True)
            
            # ================================================================================================
            # CALCULAR RANGOS DE PARÁMETROS
            # ================================================================================================
            
            st.markdown("---")
            
            if st.button("🚀 Calcular Rangos de Parámetros", type="primary", use_container_width=True):
                
                with st.spinner('🧮 Calculando rangos recomendados...'):
                    
                    # Calcular rangos
                    analisis_rangos = calcular_rangos_parametros_optimizacion(df_optimizacion)
                    
                    if not analisis_rangos['success']:
                        st.error(f"❌ Error calculando rangos: {analisis_rangos['error']}")
                        return
                    
                    # Guardar en session state
                    st.session_state.analisis_optimizacion = analisis_rangos
                    st.session_state.step_optimization = True
                    
                    st.success("✅ Rangos calculados correctamente")
            
            # ================================================================================================
            # MOSTRAR RESULTADOS SI YA SE CALCULARON
            # ================================================================================================
            
            if st.session_state.get('step_optimization', False) and 'analisis_optimizacion' in st.session_state:
                
                analisis = st.session_state.analisis_optimizacion
                
                st.markdown("---")
                st.markdown("### Resultados del Análisis")
                
                # ================================================================================================
                # MÉTRICAS DEL FILTRADO
                # ================================================================================================
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📊 Pruebas Totales", f"{analisis['total_filas_original']:,}")
                
                with col2:
                    st.metric("✅ Result > 0", f"{analisis['filas_result_positivo']:,}")
                
                with col3:
                    st.metric("📈 Result ≥ Promedio", f"{analisis['filas_finales']:,}")
                
                with col4:
                    tasa_exito = analisis['filas_finales'] / analisis['total_filas_original'] * 100
                    st.metric("🎯 Tasa de Éxito", f"{tasa_exito:.1f}%")
                
                # ================================================================================================
                # INFORMACIÓN DEL PROCESO
                # ================================================================================================
                
                st.markdown("### Proceso de Filtrado")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("** Estadísticas del Filtrado:**")
                    st.write(f"• **Pruebas originales**: {analisis['total_filas_original']:,}")
                    st.write(f"• **Con Result > 0**: {analisis['filas_result_positivo']:,}")
                    st.write(f"• **Promedio de Result**: {analisis['result_promedio']}")
                    st.write(f"• **Finales (≥ promedio)**: {analisis['filas_finales']:,}")
                
                with col2:
                    st.markdown("**🗑️ Columnas Eliminadas:**")
                    if analisis['columnas_eliminadas']:
                        for col in analisis['columnas_eliminadas'][:5]:  # Mostrar máximo 5
                            st.write(f"• {col}")
                        if len(analisis['columnas_eliminadas']) > 5:
                            st.write(f"• ... y {len(analisis['columnas_eliminadas']) - 5} más")
                    else:
                        st.write("• Ninguna columna eliminada")
                
                # ================================================================================================
                # RANGOS RECOMENDADOS - TABLA PRINCIPAL
                # ================================================================================================
                
                st.markdown("### 🎯 Rangos Recomendados para Parámetros")
                
                if analisis['rangos_parametros']:
                    
                    # Preparar datos para la tabla
                    tabla_rangos = []
                    
                    for parametro, datos in analisis['rangos_parametros'].items():
                        tabla_rangos.append({
                            'Parámetro': parametro,
                            'Tipo': datos['tipo'].title(),
                            'Rango Mínimo': datos['min'],
                            'Rango Máximo': datos['max'],
                            'Media': datos['media'],
                            'Desv. Estándar': datos['std'],
                            'Rango Completo': f"{datos['min']} - {datos['max']}"
                        })
                    
                    df_tabla_rangos = pd.DataFrame(tabla_rangos)
                    
                    # Mostrar tabla
                    st.dataframe(
                        df_tabla_rangos,
                        use_container_width=True,
                        height=min(500, len(df_tabla_rangos) * 40 + 100),
                        column_config={
                            'Parámetro': st.column_config.TextColumn('Parámetro', width="medium"),
                            'Tipo': st.column_config.TextColumn('Tipo', width="small"),
                            'Rango Mínimo': st.column_config.NumberColumn('Mín'),
                            'Rango Máximo': st.column_config.NumberColumn('Máx'),
                            'Media': st.column_config.NumberColumn('Media', format="%.4f"),
                            'Desv. Estándar': st.column_config.NumberColumn('Std', format="%.4f"),
                            'Rango Completo': st.column_config.TextColumn('Rango Completo', width="medium")
                        }
                    )
                    
                    # ================================================================================================
                    # RESUMEN EJECUTIVO
                    # ================================================================================================
                    
                    st.markdown("### 📋 Resumen Ejecutivo")
                    
                    st.success(
                        f"🎯 **{len(analisis['rangos_parametros'])} parámetros analizados** "
                        f"basados en {analisis['filas_finales']:,} pruebas exitosas "
                        f"({tasa_exito:.1f}% del total)"
                    )
                    
                    # Mostrar rangos en formato limpio
                    with st.expander("📊 Rangos Resumidos para Copy/Paste", expanded=False):
                        st.markdown("**Rangos calculados (Media ± 1 Desviación Estándar):**")
                        
                        rangos_texto = []
                        for parametro, datos in analisis['rangos_parametros'].items():
                            if datos['tipo'] == 'entero':
                                rangos_texto.append(f"• **{parametro}**: {datos['min']} - {datos['max']}")
                            else:
                                rangos_texto.append(f"• **{parametro}**: {datos['min']} - {datos['max']}")
                        
                        for linea in rangos_texto:
                            st.markdown(linea)
                    
                    # ================================================================================================
                    # GRÁFICOS DE DISTRIBUCIÓN
                    # ================================================================================================
                    
                    st.markdown("### 📊 Distribución de Parámetros")
                    
                    # Generar gráfico
                    fig_distribucion = crear_grafico_distribucion_parametros(analisis)
                    
                    if fig_distribucion:
                        st.plotly_chart(fig_distribucion, use_container_width=True)
                    else:
                        st.warning("⚠️ No se pudo generar el gráfico de distribución")
                    
                    # ================================================================================================
                    # TOP 5 MEJORES RESULTADOS
                    # ================================================================================================
                    
                    st.markdown("### 🏆 Top 5 Mejores Resultados")
                    
                    if 'mejores_resultados' in analisis and len(analisis['mejores_resultados']) > 0:
                        
                        mejores_df = analisis['mejores_resultados'].copy()
                        
                        # Redondear valores para mejor visualización
                        for col in mejores_df.select_dtypes(include=[np.number]).columns:
                            if col != 'Result':
                                mejores_df[col] = mejores_df[col].round(4)
                        
                        st.dataframe(mejores_df, use_container_width=True)
                        
                        # Mostrar el mejor resultado
                        mejor_resultado = mejores_df.iloc[0]
                        st.success(f"🥇 **Mejor resultado**: {mejor_resultado['Result']}")
                    
                    # ================================================================================================
                    # DESCARGAS
                    # ================================================================================================
                    
                    st.markdown("### 💾 Descargar Resultados")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Descarga de rangos CSV
                        csv_rangos = df_tabla_rangos.to_csv(index=False, encoding='utf-8')
                        st.download_button(
                            label="📊 Descargar Rangos (CSV)",
                            data=csv_rangos,
                            file_name=f"rangos_parametros_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        # Descarga de resumen completo
                        resumen_completo = generar_resumen_optimizacion(analisis)
                        st.download_button(
                            label="📋 Descargar Resumen (TXT)",
                            data=resumen_completo,
                            file_name=f"resumen_optimizacion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                
                else:
                    st.warning("⚠️ No se encontraron parámetros para analizar")
                
                # ================================================================================================
                # BOTÓN PARA RESET
                # ================================================================================================
                
                st.markdown("---")
                if st.button("🔄 Analizar Nuevo Archivo", type="secondary"):
                    st.session_state.step_optimization = False
                    if 'analisis_optimizacion' in st.session_state:
                        del st.session_state.analisis_optimizacion
                    st.rerun()

def sizing_tab():
    """Tab SIZING - Análisis de Position Sizing con Monte Carlo"""
    st.header("💰 SIZING - Análisis de Position Sizing")
    st.markdown("---")
    
    # Información del proceso
    st.info(
        "📋 **Proceso**: Sube el reporte HTML de backtest MT5 → "
        "Análisis Monte Carlo → Estimación de drawdown máximo probable para dimensionar posiciones"
    )
    
    # ================================================================================================
    # UPLOADER DE ARCHIVO HTML
    # ================================================================================================
    
    st.markdown("### 📁 Cargar Reporte de Backtest MT5")
    
    uploaded_html = st.file_uploader(
        "Arrastra y suelta tu archivo HTML de backtest MT5",
        type=['html', 'htm'],
        help="Archivo HTML generado por MetaTrader 5 después de ejecutar un backtest"
    )
    
    if uploaded_html is None:
        st.info("👆 Carga un archivo HTML para comenzar el análisis")
        
        # Mostrar instrucciones
        with st.expander("📖 ¿Cómo generar el archivo HTML en MT5?", expanded=False):
            st.markdown("""
            **Pasos para obtener el reporte de backtest:**
            
            1. **Ejecutar backtest en MT5**: Strategy Tester → Start
            2. **Finalizar backtest**: Esperar a que termine el proceso
            3. **Generar reporte**: Click derecho en la gráfica de resultados → "Save as Report"
            4. **Seleccionar formato**: Elegir "Web page, HTML only (*.html)"
            5. **Guardar archivo**: Dar nombre y guardar
            6. **Subir aquí**: Usar el uploader de arriba
            
            **Requisitos del archivo:**
            - Debe contener tabla de operaciones (deals)
            - Debe incluir fechas de inicio y fin del backtest
            - Formato HTML estándar de MT5
            """)
        return
    
    # ================================================================================================
    # PROCESAR ARCHIVO HTML
    # ================================================================================================
    
    if uploaded_html is not None:
        
        with st.spinner('📊 Analizando reporte de backtest...'):
            
            # Analizar archivo
            analisis_mt5 = analizar_reporte_mt5_html(uploaded_html)
            
            if not analisis_mt5['success']:
                st.error(f"❌ Error procesando archivo: {analisis_mt5['error']}")
                return
            
            # Guardar en session state
            st.session_state.analisis_mt5_sizing = analisis_mt5
            
            st.success(f"✅ Archivo procesado: {analisis_mt5['num_operaciones']:,} operaciones extraídas")
            
            # Mostrar información básica del backtest
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📊 Total Operaciones", f"{analisis_mt5['num_operaciones']:,}")
            
            with col2:
                if analisis_mt5['fecha_inicio'] and analisis_mt5['fecha_fin']:
                    duracion = (analisis_mt5['fecha_fin'] - analisis_mt5['fecha_inicio']).days
                    st.metric("📅 Duración (días)", f"{duracion:,}")
                else:
                    st.metric("📅 Duración", "No detectada")
            
            with col3:
                st.metric("💰 Retorno Total", f"{analisis_mt5['suma_total']:.2f}")
            
            # ================================================================================================
            # CÁLCULO AUTOMÁTICO DE OPERACIONES POR AÑO
            # ================================================================================================
            
            st.markdown("---")
            st.markdown("### 📈 Cálculo de Operaciones por Año")
            
            # Intentar calcular automáticamente
            if analisis_mt5['fecha_inicio'] and analisis_mt5['fecha_fin']:
                calculo_ops = calcular_operaciones_por_año(
                    analisis_mt5['fecha_inicio'],
                    analisis_mt5['fecha_fin'],
                    analisis_mt5['num_operaciones']
                )
                
                if calculo_ops['success']:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("📅 Duración del Backtest", f"{calculo_ops['duracion_años']} años")
                    
                    with col2:
                        st.metric("🔢 Operaciones/Año (Calculado)", f"{calculo_ops['operaciones_por_año']:.1f}")
                    
                    with col3:
                        st.metric("🎯 Ops/Año (Entero)", f"{calculo_ops['operaciones_por_año_entero']}")
                    
                    # Permitir ajuste manual
                    st.markdown("**Ajustar operaciones por año (opcional):**")
                    operaciones_por_año = st.number_input(
                        "Operaciones por año para simulación:",
                        min_value=1,
                        max_value=analisis_mt5['num_operaciones'],
                        value=calculo_ops['operaciones_por_año_entero'],
                        help="Número de operaciones que se ejecutarían en un año típico"
                    )
                    
                else:
                    st.warning(f"⚠️ Error calculando operaciones por año: {calculo_ops['error']}")
                    operaciones_por_año = st.number_input(
                        "Operaciones por año (manual):",
                        min_value=1,
                        max_value=analisis_mt5['num_operaciones'],
                        value=min(100, analisis_mt5['num_operaciones']),
                        help="Estima cuántas operaciones ejecutarías en un año"
                    )
            else:
                st.warning("⚠️ No se pudieron detectar fechas automáticamente")
                operaciones_por_año = st.number_input(
                    "Operaciones por año (manual):",
                    min_value=1,
                    max_value=analisis_mt5['num_operaciones'],
                    value=min(100, analisis_mt5['num_operaciones']),
                    help="Estima cuántas operaciones ejecutarías en un año"
                )
            
            # ================================================================================================
            # CONFIGURACIÓN DE SIMULACIÓN MONTE CARLO
            # ================================================================================================
            
            st.markdown("---")
            st.markdown("### ⚙️ Configuración Monte Carlo")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_simulaciones = st.selectbox(
                    "🔄 Número de Simulaciones:",
                    [1000, 2000, 5000, 10000, 20000],
                    index=3,  # 10000 por defecto
                    help="Más simulaciones = mayor precisión pero más tiempo"
                )
            
            with col2:
                capital_inicial = st.number_input(
                    "💰 Capital Inicial:",
                    min_value=100,
                    max_value=1000000,
                    value=2000,
                    step=100,
                    help="Capital inicial para cada simulación"
                )
            
            with col3:
                # Calcular tiempo estimado
                tiempo_estimado = max(1, n_simulaciones // 2000)
                st.metric("⏱️ Tiempo Estimado", f"~{tiempo_estimado} min")
            
            # Resumen de configuración
            st.info(
                f"🎯 **Configuración**: {n_simulaciones:,} simulaciones de {operaciones_por_año} operaciones "
                f"cada una, capital inicial: {capital_inicial:,}"
            )
            
            # ================================================================================================
            # BOTÓN EJECUTAR SIMULACIÓN
            # ================================================================================================
            
            if st.button("🚀 Ejecutar Simulación Monte Carlo", type="primary", use_container_width=True):
                
                try:
                    with st.spinner('🎲 Ejecutando simulaciones Monte Carlo...'):
                        
                        # Ejecutar simulación
                        simulacion_resultado = simulacion_monte_carlo_sizing(
                            analisis_mt5['retornos'],
                            operaciones_por_año,
                            n_simulaciones,
                            capital_inicial
                        )
                        
                        if not simulacion_resultado['success']:
                            st.error(f"❌ Error en simulación: {simulacion_resultado['error']}")
                            return
                        
                        # Guardar resultados en session state
                        st.session_state.simulacion_sizing = simulacion_resultado
                        st.session_state.step_sizing = True
                        
                        st.success(f"✅ Simulación completada: {n_simulaciones:,} iteraciones ejecutadas")
                        
                except Exception as e:
                    st.error(f"❌ Error ejecutando simulación: {str(e)}")
                    return
            
            # ================================================================================================
            # MOSTRAR RESULTADOS SI YA SE COMPLETÓ
            # ================================================================================================
            
            if st.session_state.get('step_sizing', False) and 'simulacion_sizing' in st.session_state:
                
                simulacion_resultado = st.session_state.simulacion_sizing
                
                st.markdown("---")
                st.markdown("### 📊 Resultados de la Simulación")
                
                # ================================================================================================
                # MÉTRICAS PRINCIPALES
                # ================================================================================================
                
                st.markdown("#### 💰 Métricas de Drawdown Máximo")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📊 Media", f"{simulacion_resultado['dd_mean']:.2f}")
                
                with col2:
                    st.metric("📈 Mediana", f"{simulacion_resultado['dd_median']:.2f}")
                
                with col3:
                    st.metric("📉 Percentil 95%", f"{simulacion_resultado['dd_percentiles'][95]:.2f}")
                
                with col4:
                    st.metric("🔴 Percentil 99%", f"{simulacion_resultado['dd_percentiles'][99]:.2f}")
                
                # ================================================================================================
                # GRÁFICOS PRINCIPALES
                # ================================================================================================
                
                st.markdown("#### 📈 Gráficos de Distribución")
                
                # Tabs para diferentes gráficos
                graph_tab1, graph_tab2, graph_tab3 = st.tabs([
                    "📉 Distribución Drawdown",
                    "💰 Distribución Profit", 
                    "📊 Drawdown Histórico"
                ])
                
                with graph_tab1:
                    fig_dd = crear_grafico_distribucion_monte_carlo(simulacion_resultado, 'drawdown')
                    st.plotly_chart(fig_dd, use_container_width=True)
                
                with graph_tab2:
                    fig_profit = crear_grafico_distribucion_monte_carlo(simulacion_resultado, 'profit')
                    st.plotly_chart(fig_profit, use_container_width=True)
                
                with graph_tab3:
                    # Calcular drawdown del histórico completo
                    dd_historico = calcular_maxdd(analisis_mt5['retornos'], capital_inicial)
                    fig_historico = crear_grafico_drawdown_historico(dd_historico)
                    st.plotly_chart(fig_historico, use_container_width=True)
                
                # ================================================================================================
                # TABLA DE PERCENTILES
                # ================================================================================================
                
                st.markdown("#### 📋 Tabla de Percentiles")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📉 Drawdown Máximo:**")
                    percentiles_dd_data = []
                    for p, val in simulacion_resultado['dd_percentiles'].items():
                        percentiles_dd_data.append({
                            'Percentil': f'{p}%',
                            'Drawdown': f'{val:.2f}',
                            'Probabilidad': f'{100-p}%'
                        })
                    
                    df_percentiles_dd = pd.DataFrame(percentiles_dd_data)
                    st.dataframe(df_percentiles_dd, use_container_width=True, hide_index=True)
                
                with col2:
                    st.markdown("**💰 Profit Anual:**")
                    percentiles_profit_data = []
                    for p, val in simulacion_resultado['profit_percentiles'].items():
                        percentiles_profit_data.append({
                            'Percentil': f'{p}%',
                            'Profit': f'{val:.2f}',
                            'Probabilidad': f'{p}%'
                        })
                    
                    df_percentiles_profit = pd.DataFrame(percentiles_profit_data)
                    st.dataframe(df_percentiles_profit, use_container_width=True, hide_index=True)
                
                # ================================================================================================
                # RECOMENDACIONES DE POSITION SIZING
                # ================================================================================================
                
                st.markdown("#### 🎯 Recomendaciones de Position Sizing")
                
                p95_dd = simulacion_resultado['dd_percentiles'][95]
                p99_dd = simulacion_resultado['dd_percentiles'][99]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Para 95% de confianza:**")
                    st.success(f"💰 Preparar capital para DD máximo: **{p95_dd:.2f}**")
                    if capital_inicial > 0 and p95_dd > 0:
                        ratio_95 = capital_inicial / p95_dd
                        st.info(f"📈 Ratio de seguridad: **{ratio_95:.2f}x**")
                
                with col2:
                    st.markdown("**📊 Para 99% de confianza:**")
                    st.error(f"💰 Preparar capital para DD máximo: **{p99_dd:.2f}**")
                    if capital_inicial > 0 and p99_dd > 0:
                        ratio_99 = capital_inicial / p99_dd
                        st.info(f"📈 Ratio de seguridad: **{ratio_99:.2f}x**")
                
                # Interpretación adicional
                st.markdown("**💡 Interpretación:**")
                st.write(f"• En el **95%** de los casos, el drawdown máximo será ≤ {p95_dd:.2f}")
                st.write(f"• En el **99%** de los casos, el drawdown máximo será ≤ {p99_dd:.2f}")
                st.write(f"• En el **5%** de los casos, el drawdown podría superar {p95_dd:.2f}")
                st.write(f"• En el **1%** de los casos, el drawdown podría superar {p99_dd:.2f}")
                
                # ================================================================================================
                # COMPARACIÓN CON HISTÓRICO
                # ================================================================================================
                
                st.markdown("#### 🔍 Comparación con Histórico")
                
                dd_historico = calcular_maxdd(analisis_mt5['retornos'], capital_inicial)
                dd_hist_max = dd_historico['max_dd_abs']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("📊 DD Histórico Real", f"{dd_hist_max:.2f}")
                
                with col2:
                    percentil_historico = np.percentile(simulacion_resultado['max_dd_list'], 
                                                      [p for p, v in simulacion_resultado['dd_percentiles'].items() 
                                                       if v >= dd_hist_max])
                    percentil_hist = min(percentil_historico) if len(percentil_historico) > 0 else 100
                    st.metric("📈 Percentil del Histórico", f"~{percentil_hist:.0f}%")
                
                with col3:
                    comparacion = "Normal" if dd_hist_max <= p95_dd else ("Alto" if dd_hist_max <= p99_dd else "Extremo")
                    color = "🟢" if comparacion == "Normal" else ("🟡" if comparacion == "Alto" else "🔴")
                    st.metric("🎯 Nivel Histórico", f"{color} {comparacion}")
                
                # ================================================================================================
                # DESCARGA DE RESULTADOS
                # ================================================================================================
                
                st.markdown("---")
                st.markdown("### 💾 Descargar Resultados")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Generar reporte completo
                    reporte_completo = generar_reporte_sizing(analisis_mt5, simulacion_resultado)
                    
                    st.download_button(
                        label="📋 Descargar Reporte Completo",
                        data=reporte_completo,
                        file_name=f"reporte_sizing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                
                with col2:
                    # Generar CSV con resultados detallados
                    df_resultados = pd.DataFrame({
                        'Simulacion': range(1, len(simulacion_resultado['max_dd_list']) + 1),
                        'Drawdown_Maximo': simulacion_resultado['max_dd_list'],
                        'Profit_Total': simulacion_resultado['profits_list']
                    })
                    
                    csv_resultados = df_resultados.to_csv(index=False)
                    
                    st.download_button(
                        label="📊 Descargar Datos CSV",
                        data=csv_resultados,
                        file_name=f"simulaciones_sizing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                # ================================================================================================
                # BOTÓN PARA NUEVA SIMULACIÓN
                # ================================================================================================
                
                st.markdown("---")
                if st.button("🔄 Nueva Simulación", type="secondary"):
                    st.session_state.step_sizing = False
                    if 'simulacion_sizing' in st.session_state:
                        del st.session_state.simulacion_sizing
                    st.rerun()

# ================================================================================================
# FUNCIÓN PRINCIPAL
# ================================================================================================

def main():
    """Aplicación principal"""
    initialize_session_state()
    
    # Encabezado con imagen BFUNDED
    import os
    bfunded_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../recursos/BFUNDED.png'))
    col_logo, col_title = st.columns([1, 6])
    with col_logo:
        st.image(bfunded_path, width=480)
    with col_title:
        st.title("Análisis de reportes de optimización Metatrader 5")

    # Solo mostrar la interfaz de Optimization
    optimization_tab()
# ================================================================================================
# EJECUCIÓN
# ================================================================================================

if __name__ == "__main__":
    main()

