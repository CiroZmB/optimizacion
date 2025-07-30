import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.feature_selection import mutual_info_regression
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import xml.etree.ElementTree as ET
import warnings
warnings.filterwarnings('ignore')

def clean_data(df):
   """
   Limpia y formatea un DataFrame de datos forex/indicadores técnicos.
   
   Parámetros:
   -----------
   df : pandas.DataFrame
       DataFrame con columnas: Date, Open, High, Low, Close + indicadores
       
   Retorna:
   --------
   pandas.DataFrame
       DataFrame limpio con fechas como índice
       
   Funciones:
   ----------
   - Elimina filas con valores extremos en columnas de indicadores (6+)
   - Convierte columna Date a formato fecha sin hora
   - Establece Date como índice del DataFrame
   """

   # Limpiar valores extremos en columnas de indicadores (desde columna 6)
   df = df[~df.iloc[:, 5:].astype(str).apply(lambda x: x.str.contains('1797693134862315', na=False)).any(axis=1)]

   # Formatear fechas y establecer como índice
   df['Date'] = pd.to_datetime(df['Date']).dt.date
   df.set_index('Date', inplace=True)
   
   return df

def detect_timeframe(df):
    """
    Detecta automáticamente la frecuencia temporal del DataFrame
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame con fechas como índice
        
    Retorna:
    --------
    str : Descripción de la frecuencia detectada
    """
    try:
        # Calcular diferencias entre fechas consecutivas
        time_diffs = pd.Series(df.index).diff().dropna()
        
        # Obtener la diferencia más común
        most_common_diff = time_diffs.mode()[0]
        
        # Determinar la frecuencia
        if most_common_diff.days == 1:
            return "Diario (1D)"
        elif most_common_diff.days == 7:
            return "Semanal (1W)"
        elif most_common_diff.days >= 28 and most_common_diff.days <= 31:
            return "Mensual (1M)"
        elif most_common_diff.total_seconds() == 3600:  # 1 hora
            return "Horario (1H)"
        elif most_common_diff.total_seconds() == 14400:  # 4 horas
            return "4 Horas (4H)"
        elif most_common_diff.total_seconds() == 900:  # 15 minutos
            return "15 Minutos (15M)"
        elif most_common_diff.total_seconds() == 300:  # 5 minutos
            return "5 Minutos (5M)"
        elif most_common_diff.total_seconds() == 60:  # 1 minuto
            return "1 Minuto (1M)"
        else:
            return f"Personalizado (~{most_common_diff})"
    except:
        return "No detectado"

def change_target(df, days_ahead=1):
    """
    Añade una columna 'Target' con el retorno a N días vista.
    
    Target = (Open[t+N+1] - Open[t+1]) / Open[t+1]
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame con columna 'Open' y fechas como índice
    days_ahead : int, default=1
        Número de días vista para el cálculo del retorno
        
    Retorna:
    --------
    pandas.DataFrame
        DataFrame con nueva columna 'Target'
        
    Ejemplos:
    ---------
    days_ahead=1: Target[hoy] = (Open[pasado mañana] - Open[mañana]) / Open[mañana]
    days_ahead=2: Target[hoy] = (Open[+3 días] - Open[mañana]) / Open[mañana]
    days_ahead=5: Target[hoy] = (Open[+6 días] - Open[mañana]) / Open[mañana]
    
    Nota:
    -----
    - Las últimas (days_ahead + 1) filas tendrán Target = NaN
    """

    
    # Open de mañana (siempre t+1)
    open_tomorrow = df['Open'].shift(-1)
    
    # Open de N+1 días después
    open_target_day = df['Open'].shift(-(days_ahead + 1))
    
    # Target = (Open[t+N+1] - Open[t+1]) / Open[t+1]
    df['Target'] = (open_target_day - open_tomorrow) / open_tomorrow
    
    return df

def ibs_target(df, days_ahead=1):
    """
    Añade una columna 'IBS_Target' con el IBS extendido a N días vista.
    
    IBS_Target = (Close_final - Low_min_período) / (High_max_período - Low_min_período)
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame con columnas 'High', 'Low', 'Close' y fechas como índice
    days_ahead : int, default=1
        Número de días del período a considerar
        
    Retorna:
    --------
    pandas.DataFrame
        DataFrame con nueva columna 'IBS_Target'
        
    Ejemplos:
    ---------
    days_ahead=1: Período = solo mañana
                  IBS = IBS tradicional de mañana
                  
    days_ahead=2: Período = mañana + pasado mañana  
                  Close = cierre de pasado mañana
                  High = máximo entre mañana y pasado mañana
                  Low = mínimo entre mañana y pasado mañana
                  
    days_ahead=5: Período = próximos 5 días
                  Close = cierre del día 5
                  High/Low = máximo/mínimo de los 5 días
    
    Cálculo específico:
    ------------------
    Para days_ahead=2 en fecha t:
    - Período: días t+1 y t+2 (mañana y pasado mañana)
    - Close_final: Close[t+2] (cierre de pasado mañana)
    - High_max: max(High[t+1], High[t+2])
    - Low_min: min(Low[t+1], Low[t+2])
    - IBS_Target = (Close[t+2] - Low_min) / (High_max - Low_min)
    
    Nota:
    -----
    - Las últimas days_ahead filas tendrán IBS_Target = NaN
    """
    ibs_target = []
    
    for i in range(len(df)):
        # Verificar si tenemos suficientes datos futuros
        if i + days_ahead >= len(df):
            ibs_target.append(np.nan)
            continue
        
        # Período: los próximos N días (desde t+1 hasta t+N)
        start_idx = i + 1  # Mañana
        end_idx = i + days_ahead  # Último día del período
        
        # Obtener datos del período
        period_high = df['High'].iloc[start_idx:end_idx + 1]
        period_low = df['Low'].iloc[start_idx:end_idx + 1]
        final_close = df['Close'].iloc[end_idx]  # Close del último día
        
        # Calcular máximo y mínimo del período
        max_high = period_high.max()
        min_low = period_low.min()
        
        # Calcular IBS extendido
        if max_high == min_low:  # Evitar división por cero
            ibs_value = 0.5
        else:
            ibs_value = (final_close - min_low) / (max_high - min_low)
        
        ibs_target.append(ibs_value)
    
    df['Target'] = ibs_target
    
    return df

def split_data(df, mode='classic', train_ratio=0.7, start_date=None, end_date=None, train_start=None, train_end=None):
   """
   Particiona el DataFrame en conjuntos de entrenamiento y prueba.
   
   Parámetros:
   -----------
   df : pandas.DataFrame
       DataFrame con fechas como índice y columna 'Target'
   mode : str, default='classic'
       Modo de partición:
       - 'classic': train = datos antiguos, test = datos recientes
       - 'inverted': train = datos recientes, test = datos antiguos  
       - 'free': train = período personalizado, test = resto
   train_ratio : float, default=0.7
       Porcentaje de datos para entrenamiento (solo para 'classic' e 'inverted')
       Puede ser 1.0 para usar todos los datos como train
   start_date : str or datetime, default=None
       Fecha inicial para filtrar datos (formato 'YYYY-MM-DD')
   end_date : str or datetime, default=None
       Fecha final para filtrar datos (formato 'YYYY-MM-DD')
   train_start : str or datetime, default=None
       Fecha inicial del conjunto de entrenamiento (solo para mode='free')
   train_end : str or datetime, default=None
       Fecha final del conjunto de entrenamiento (solo para mode='free')
       
   Retorna:
   --------
   tuple : (train_df, test_df)
       DataFrames de entrenamiento y prueba con todas las columnas
       
   Ejemplos:
   ---------
   # Partición clásica con 80% train
   train_df, test_df = partition_data(df, mode='classic', train_ratio=0.8)
   
   # Usar todos los datos como train
   train_df, test_df = partition_data(df, mode='classic', train_ratio=1.0)
   
   # Partición libre con fechas personalizadas
   train_df, test_df = partition_data(df, mode='free',
                                     train_start='2015-01-01',
                                     train_end='2018-12-31')
   """
   # Verificar que existe la columna Target
   if 'Target' not in df.columns:
       raise ValueError("No se encontró la columna 'Target' en el DataFrame")
   
   # Verificar train_ratio válido
   if not 0 < train_ratio <= 1:
       raise ValueError("train_ratio debe estar entre 0 y 1 (inclusive)")
   
   # Filtrar por rango de fechas si se especifica
   if start_date is not None:
       start_date = pd.to_datetime(start_date).date()
       df = df[df.index >= start_date]
   if end_date is not None:
       end_date = pd.to_datetime(end_date).date()
       df = df[df.index <= end_date]
   
   # Eliminar filas con NaN en Target
   df = df.dropna(subset=['Target'])
   
   # Verificar que hay datos después del filtrado
   if len(df) == 0:
       raise ValueError("No hay datos válidos después del filtrado")
   
   # Realizar partición según el modo
   if mode == 'classic':
       # Classic: train = datos antiguos, test = datos recientes
       split_idx = int(len(df) * train_ratio)
       
       train_df = df.iloc[:split_idx].copy()
       test_df = df.iloc[split_idx:].copy()
       
   elif mode == 'inverted':
       # Inverted: train = datos recientes, test = datos antiguos
       split_idx = int(len(df) * (1 - train_ratio))
       
       train_df = df.iloc[split_idx:].copy()
       test_df = df.iloc[:split_idx].copy()
       
   elif mode == 'free':
       # Free: train = período personalizado, test = resto
       if train_start is None or train_end is None:
           raise ValueError("Para mode='free' se requieren train_start y train_end")
       
       train_start = pd.to_datetime(train_start).date()
       train_end = pd.to_datetime(train_end).date()
       
       # Crear máscaras
       train_mask = (df.index >= train_start) & (df.index <= train_end)
       test_mask = ~train_mask
       
       train_df = df[train_mask].copy()
       test_df = df[test_mask].copy()
       
   else:
       raise ValueError("mode debe ser 'classic', 'inverted' o 'free'")
   
   return train_df, test_df

def calculate_feature_metrics(df, metric='pearson'):
    """
    Calcula métricas de relación entre features y target
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset con features y columna 'Target'
    metric : str
        Métrica a calcular: 'pearson', 'spearman', 'kendall', 'mutual_info'
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame con índice=nombres de features, columna=valores de métrica
    """
    
    # Validar que existe la columna Target
    if 'Target' not in df.columns:
        raise ValueError("El DataFrame debe contener una columna llamada 'Target'")
    
    # Hacer copia para no modificar el original
    df_work = df.copy()
    
    # Eliminar filas donde Target es NaN
    df_work = df_work.dropna(subset=['Target'])
    
    if len(df_work) == 0:
        raise ValueError("No hay datos válidos después de eliminar NaN en Target")
    
    # Columnas a ignorar
    ignore_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Target']
    
    # Obtener features (todas las columnas excepto las ignoradas)
    feature_cols = [col for col in df_work.columns if col not in ignore_cols]
    
    if len(feature_cols) == 0:
        raise ValueError("No hay features válidas para calcular")
    
    # Target
    target = df_work['Target'].values
    
    # Diccionario para almacenar resultados
    results = {}
    
    for feature in feature_cols:
        try:
            # Obtener feature y eliminar NaN adicionales
            feature_data = df_work[feature].values
            
            # Crear máscara para valores válidos en ambas variables
            valid_mask = ~(np.isnan(feature_data) | np.isnan(target))
            
            if np.sum(valid_mask) < 2:  # Necesitamos al menos 2 puntos
                results[feature] = 0.0
                continue
                
            feature_clean = feature_data[valid_mask]
            target_clean = target[valid_mask]
            
            # Verificar varianza
            if np.var(feature_clean) == 0 or np.var(target_clean) == 0:
                results[feature] = 0.0
                continue
            
            # Calcular métrica según elección
            if metric == 'pearson':
                corr, p_value = pearsonr(feature_clean, target_clean)
                results[feature] = abs(corr)  # Valor absoluto para ranking
                
            elif metric == 'spearman':
                corr, p_value = spearmanr(feature_clean, target_clean)
                results[feature] = abs(corr)
                
            elif metric == 'kendall':
                corr, p_value = kendalltau(feature_clean, target_clean)
                results[feature] = abs(corr)
                
            elif metric == 'mutual_info':
                # Mutual Information (siempre positiva)
                mi = mutual_info_regression(feature_clean.reshape(-1, 1), target_clean, random_state=42)
                results[feature] = mi[0]
                
            else:
                raise ValueError(f"Métrica '{metric}' no reconocida. Opciones: 'pearson', 'spearman', 'kendall', 'mutual_info', 'distance_corr'")
                
        except Exception as e:
            print(f"Error calculando {metric} para {feature}: {str(e)}")
            results[feature] = 0.0
    
    # Crear DataFrame resultado
    result_df = pd.DataFrame.from_dict(results, orient='index', columns=[metric])
    
    # Ordenar por valor descendente
    result_df = result_df.sort_values(by=metric, ascending=False)
    
    return result_df

def remove_correlated_features(df_train, feature_metrics_df=None, threshold=0.95, n_features=None, random_removal=False):
    """
    Elimina features altamente correlacionadas manteniendo la de mejor métrica vs target o aleatoriamente
    
    Parameters:
    -----------
    df_train : pandas.DataFrame
        Dataset de entrenamiento con todas las features
    feature_metrics_df : pandas.DataFrame or None
        DataFrame con índice=features y una columna con valores de métrica vs target
        (resultado de calculate_feature_metrics). Si None, se requiere random_removal=True
    threshold : float
        Umbral de correlación para eliminar features (default: 0.95)
    n_features : int or None
        Número máximo de features a devolver. Si None, devuelve todas las que pasan el filtro
    random_removal : bool
        Si True, elimina aleatoriamente entre features correlacionadas.
        Si False, elimina la de menor métrica (requiere feature_metrics_df)
    
    Returns:
    --------
    pandas.DataFrame
        Si random_removal=False: feature_metrics_df filtrado con las features seleccionadas
        Si random_removal=True: DataFrame con índice=features y columna 'selected'=True
    """
    import random
    
    # Validaciones iniciales
    if not random_removal and feature_metrics_df is None:
        raise ValueError("feature_metrics_df es requerido cuando random_removal=False")
    
    if random_removal and feature_metrics_df is not None and feature_metrics_df.shape[1] != 1:
        raise ValueError("feature_metrics_df debe tener exactamente una columna con los valores de métrica")
    
    # Obtener features disponibles
    ignore_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Target']
    if feature_metrics_df is not None:
        available_features = list(set(feature_metrics_df.index) & set(df_train.columns))
    else:
        available_features = [col for col in df_train.columns if col not in ignore_cols]
    
    if len(available_features) == 0:
        raise ValueError("No hay features disponibles para procesar")
    
    # Crear subset con solo las features disponibles
    df_features = df_train[available_features].copy()
    
    # Eliminar filas con NaN para cálculo de correlación
    df_clean = df_features.dropna()
    
    if len(df_clean) < 2:
        raise ValueError("Datos insuficientes para calcular correlaciones")
    
    # Calcular matriz de correlación
    corr_matrix = df_clean.corr().abs()  # Valor absoluto para considerar correlaciones negativas
    
    # Encontrar pares correlacionados y features a eliminar
    features_to_remove = set()
    
    # Iterar sobre todas las combinaciones de features
    for i, feature1 in enumerate(available_features):
        for feature2 in available_features[i+1:]:
            correlation = corr_matrix.loc[feature1, feature2]
            
            if correlation > threshold:
                if random_removal:
                    # Eliminación aleatoria
                    feature_to_remove = random.choice([feature1, feature2])
                    features_to_remove.add(feature_to_remove)
                else:
                    # Eliminación por métrica (comportamiento original)
                    metric_column = feature_metrics_df.columns[0]
                    metric1 = feature_metrics_df.loc[feature1, metric_column]
                    metric2 = feature_metrics_df.loc[feature2, metric_column]
                    
                    # Determinar cuál eliminar (la de menor métrica)
                    if metric1 >= metric2:
                        features_to_remove.add(feature2)
                    else:
                        features_to_remove.add(feature1)
    
    # Crear lista de features a mantener
    features_to_keep = [f for f in available_features if f not in features_to_remove]
    
    if random_removal:
        # Para eliminación aleatoria, devolver DataFrame simple con features seleccionadas
        result_df = pd.DataFrame(
            index=features_to_keep, 
            data={'selected': [True] * len(features_to_keep)}
        )
        
        # No aplicar límite n_features en modo random (no hay métrica para ordenar)
        return result_df
    
    else:
        # Para eliminación por métrica, devolver feature_metrics_df filtrado
        filtered_metrics = feature_metrics_df.loc[features_to_keep].copy()
        
        # Ordenar por métrica descendente
        metric_column = feature_metrics_df.columns[0]
        filtered_metrics = filtered_metrics.sort_values(by=metric_column, ascending=False)
        
        # Aplicar límite de n_features si se especifica
        if n_features is not None and n_features > 0:
            filtered_metrics = filtered_metrics.head(n_features)
        
        return filtered_metrics


# ═══════════════════════════════════════════════════════════════
# EQUITY CURVES
# ═══════════════════════════════════════════════════════════════

def generar_equity_curves(df_completo, reglas_seleccionadas):
    """
    Genera curvas de equity para las reglas seleccionadas (VERSIÓN CON DEBUG)
    
    Parameters:
    -----------
    df_completo : pandas.DataFrame
        DataFrame completo con todas las columnas y Target
    reglas_seleccionadas : list
        Lista de diccionarios con reglas del ENSEMBLE
        
    Returns:
    --------
    dict
        Diccionario con equity curves de cada regla
    """
    
    print(f"🔍 DEBUG - generar_equity_curves:")
    print(f"   Reglas recibidas: {len(reglas_seleccionadas)}")
    
    # Limpiar datos
    df_clean = df_completo.dropna(subset=['Target']).sort_index()
    
    if len(df_clean) == 0:
        raise ValueError("No hay datos válidos para generar equity curves")
    
    print(f"   Dataset limpio: {len(df_clean)} filas")
    print(f"   Columnas disponibles: {list(df_clean.columns)}")
    
    equity_data = {}
    
    # Inicializar equity curve del benchmark (buy & hold)
    equity_data['benchmark'] = {
        'fechas': df_clean.index,
        'equity': df_clean['Target'].cumsum(),
        'descripcion': 'Buy & Hold',
        'tipo': 'BENCHMARK',
        'operaciones': len(df_clean),
        'rendimiento_total': df_clean['Target'].sum()
    }
    
    print(f"   ✅ Benchmark creado: {equity_data['benchmark']['rendimiento_total']:.6f}")
    
    # Generar equity curve para cada regla
    for i, regla in enumerate(reglas_seleccionadas):
        print(f"\n   📊 Procesando regla {i+1}/{len(reglas_seleccionadas)}: {regla['descripcion']} ({regla['tipo']})")
        print(f"      Regla: {regla['regla_compuesta'][:100]}...")
        
        try:
            # Normalizar regla (por si acaso)
            regla_normalizada = regla['regla_compuesta'].replace(' AND ', ' and ').replace(' OR ', ' or ')
            print(f"      Regla normalizada: {regla_normalizada[:100]}...")
            
            # Aplicar regla
            matches = df_clean.query(regla_normalizada)
            print(f"      Matches encontrados: {len(matches)}")
            
            if len(matches) == 0:
                print(f"      ⚠️ Sin matches para {regla['descripcion']}")
                continue
            
            # Definir multiplicador según tipo
            M = 1 if regla['tipo'] == 'LONG' else -1
            print(f"      Multiplicador M: {M}")
            
            # Crear serie temporal de operaciones
            operaciones_serie = pd.Series(0.0, index=df_clean.index)
            operaciones_serie.loc[matches.index] = M * matches['Target']
            
            # Calcular equity curve acumulada
            equity_curve = operaciones_serie.cumsum()
            
            # Calcular rendimiento total
            rendimiento_total = operaciones_serie.sum()
            print(f"      Rendimiento total: {rendimiento_total:.6f}")
            
            # Guardar datos
            equity_data[regla['descripcion']] = {
                'fechas': df_clean.index,
                'equity': equity_curve,
                'descripcion': regla['descripcion'],
                'tipo': regla['tipo'],
                'operaciones': len(matches),
                'rendimiento_total': rendimiento_total,
                'ratio_estabilidad': regla.get('ratio_estabilidad', 0)
            }
            
            print(f"      ✅ Regla procesada exitosamente")
            
        except Exception as e:
            print(f"      ❌ Error procesando regla {regla['descripcion']}: {str(e)}")
            continue
    
    print(f"\n🎯 RESUMEN:")
    print(f"   Total equity curves generadas: {len(equity_data)}")
    for name, data in equity_data.items():
        print(f"   - {name} ({data['tipo']}): {data['operaciones']} ops, {data['rendimiento_total']:.6f} rend")
    
    return equity_data

def crear_grafico_equity_individual(equity_data):
    """
    Crea gráfico con equity curves individuales
    
    Parameters:
    -----------
    equity_data : dict
        Diccionario resultado de generar_equity_curves
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Figura de Plotly con equity curves
    """
    
    fig = go.Figure()
    
    # Colores para diferentes tipos
    colors = {
        'LONG': '#2E8B57',     # Verde
        'SHORT': '#DC143C',    # Rojo
        'BENCHMARK': '#808080' # Gris
    }
    
    # Añadir cada equity curve
    for regla_name, data in equity_data.items():
        
        color = colors.get(data['tipo'], '#1f77b4')
        
        # Estilo de línea
        if data['tipo'] == 'BENCHMARK':
            line_dash = 'dash'
            line_width = 2
        else:
            line_dash = 'solid'
            line_width = 1.5
        
        fig.add_trace(go.Scatter(
            x=data['fechas'],
            y=data['equity'],
            mode='lines',
            name=f"{data['descripcion']} ({data['tipo']})",
            line=dict(color=color, dash=line_dash, width=line_width),
            hovertemplate=(
                f'<b>{data["descripcion"]}</b><br>' +
                'Fecha: %{x}<br>' +
                'Equity: %{y:.6f}<br>' +
                f'Tipo: {data["tipo"]}<br>' +
                f'Operaciones: {data["operaciones"]}<br>' +
                f'Rendimiento Total: {data["rendimiento_total"]:.6f}<extra></extra>'
            )
        ))
    
    # Layout
    fig.update_layout(
        title=dict(
            text='📈 Equity Curves - Reglas Individuales',
            x=0.5,
            font=dict(size=20)
        ),
        xaxis_title='Fecha',
        yaxis_title='Equity Acumulada',
        hovermode='x unified',
        template='plotly_white',
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        )
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

def crear_grafico_equity_combinado(equity_data):
    """
    Crea gráfico del portfolio combinado
    
    Parameters:
    -----------
    equity_data : dict
        Diccionario resultado de generar_equity_curves
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Figura de Plotly con portfolio combinado
    """
    
    # Excluir benchmark del portfolio
    reglas_trading = {k: v for k, v in equity_data.items() if v['tipo'] != 'BENCHMARK'}
    
    if len(reglas_trading) == 0:
        # Crear figura vacía
        fig = go.Figure()
        fig.add_annotation(
            text="No hay reglas válidas para portfolio combinado",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Obtener fechas de referencia
    fechas_ref = list(equity_data.values())[0]['fechas']
    
    # Calcular portfolio combinado (promedio ponderado por ratio de estabilidad)
    portfolio_equity = pd.Series(0.0, index=fechas_ref)
    total_weight = 0
    
    for regla_name, data in reglas_trading.items():
        # Peso basado en ratio de estabilidad (normalizado)
        weight = max(0.1, data.get('ratio_estabilidad', 1.0))  # Mínimo 0.1
        portfolio_equity += data['equity'] * weight
        total_weight += weight
    
    # Normalizar por peso total
    if total_weight > 0:
        portfolio_equity = portfolio_equity / total_weight
    
    # Crear figura
    fig = go.Figure()
    
    # Añadir portfolio combinado
    fig.add_trace(go.Scatter(
        x=fechas_ref,
        y=portfolio_equity,
        mode='lines',
        name='Portfolio Combinado',
        line=dict(color='#FF6347', width=3),
        hovertemplate=(
            '<b>Portfolio Combinado</b><br>' +
            'Fecha: %{x}<br>' +
            'Equity: %{y:.6f}<br>' +
            f'Reglas: {len(reglas_trading)}<extra></extra>'
        )
    ))
    
    # Añadir benchmark para comparación
    if 'benchmark' in equity_data:
        benchmark_data = equity_data['benchmark']
        fig.add_trace(go.Scatter(
            x=benchmark_data['fechas'],
            y=benchmark_data['equity'],
            mode='lines',
            name='Benchmark (Buy & Hold)',
            line=dict(color='#808080', dash='dash', width=2),
            hovertemplate=(
                '<b>Benchmark</b><br>' +
                'Fecha: %{x}<br>' +
                'Equity: %{y:.6f}<extra></extra>'
            )
        ))
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f'📈 Portfolio Combinado vs Benchmark ({len(reglas_trading)} reglas)',
            x=0.5,
            font=dict(size=20)
        ),
        xaxis_title='Fecha',
        yaxis_title='Equity Acumulada',
        hovermode='x unified',
        template='plotly_white',
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        )
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

def calcular_metricas_equity(equity_data):
    """
    Calcula métricas de performance para las equity curves
    
    Parameters:
    -----------
    equity_data : dict
        Diccionario resultado de generar_equity_curves
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame con métricas de cada regla
    """
    
    metricas = []
    
    for regla_name, data in equity_data.items():
        equity_series = data['equity']
        
        # Calcular retornos diarios
        retornos = equity_series.diff().dropna()
        
        if len(retornos) == 0:
            continue
        
        # Métricas básicas
        rendimiento_total = data['rendimiento_total']
        operaciones = data['operaciones']
        rendimiento_por_operacion = rendimiento_total / operaciones if operaciones > 0 else 0
        
        # Métricas de riesgo
        volatilidad = retornos.std() * np.sqrt(252) if len(retornos) > 1 else 0  # Anualizada
        
        # Sharpe ratio (sin risk-free rate)
        sharpe_ratio = (retornos.mean() / retornos.std()) * np.sqrt(252) if retornos.std() > 0 else 0
        
        # Drawdown máximo
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_drawdown = drawdown.min()
        
        # Ratio Calmar
        calmar_ratio = (rendimiento_total / abs(max_drawdown)) if max_drawdown != 0 else 0
        
        # Win rate (porcentaje de operaciones positivas)
        if data['tipo'] != 'BENCHMARK':
            # Recalcular operaciones para win rate
            try:
                df_temp = pd.DataFrame({'Target': [0] * len(data['fechas'])}, index=data['fechas'])
                matches_temp = df_temp.query(regla_name.split('(')[0].strip())  # Simplificado
                operaciones_positivas = 0  # Simplificado por ahora
                win_rate = 0  # Simplificado por ahora
            except:
                win_rate = 0
        else:
            # Para benchmark, calcular días positivos vs negativos
            dias_positivos = (retornos > 0).sum()
            win_rate = dias_positivos / len(retornos) if len(retornos) > 0 else 0
        
        # Compilar métricas
        metricas.append({
            'Regla': data['descripcion'],
            'Tipo': data['tipo'],
            'Rendimiento Total': f"{rendimiento_total:.6f}",
            'Operaciones': operaciones,
            'Rend./Operación': f"{rendimiento_por_operacion:.6f}",
            'Volatilidad (anual)': f"{volatilidad:.4f}",
            'Sharpe Ratio': f"{sharpe_ratio:.3f}",
            'Max Drawdown': f"{max_drawdown:.4f}",
            'Calmar Ratio': f"{calmar_ratio:.3f}",
            'Win Rate': f"{win_rate:.1%}",
            'Ratio Estabilidad': f"{data.get('ratio_estabilidad', 0):.3f}"
        })
    
    return pd.DataFrame(metricas)

# ═══════════════════════════════════════════════════════════════
# FUNCIONES DE DESCARGA - AÑADIR A utils.py
# ═══════════════════════════════════════════════════════════════

def preparar_descarga_reglas_long(top_long_df):
    """
    Prepara las reglas LONG para descarga en formato CSV
    
    Parameters:
    -----------
    top_long_df : pandas.DataFrame
        DataFrame con reglas LONG del ENSEMBLE
        
    Returns:
    --------
    str
        String en formato CSV listo para descarga
    """
    
    if len(top_long_df) == 0:
        return "No hay reglas LONG disponibles para descargar"
    
    # Crear DataFrame para descarga con información relevante
    download_df = top_long_df.copy()
    
    # Seleccionar y renombrar columnas
    columns_download = {
        'descripcion': 'Variables',
        'tipo': 'Tipo',
        'regla_compuesta': 'Regla_Compuesta',
        'rendimiento_promedio': 'Rendimiento_Promedio',
        'std_rendimientos': 'Desviacion_Estandar',
        'ratio_estabilidad': 'Ratio_Estabilidad',
        'rendimiento_total_original': 'Rendimiento_Total_Original',
        'operaciones_total_original': 'Operaciones_Total'
    }
    
    # Filtrar solo las columnas que existen
    available_columns = {k: v for k, v in columns_download.items() if k in download_df.columns}
    
    download_df = download_df[list(available_columns.keys())].copy()
    download_df = download_df.rename(columns=available_columns)
    
    # Redondear valores numéricos para mejor legibilidad
    numeric_columns = download_df.select_dtypes(include=[np.number]).columns
    download_df[numeric_columns] = download_df[numeric_columns].round(6)
    
    # Convertir a CSV
    csv_string = download_df.to_csv(index=False, encoding='utf-8')
    
    return csv_string

def preparar_descarga_reglas_short(top_short_df):
    """
    Prepara las reglas SHORT para descarga en formato CSV
    
    Parameters:
    -----------
    top_short_df : pandas.DataFrame
        DataFrame con reglas SHORT del ENSEMBLE
        
    Returns:
    --------
    str
        String en formato CSV listo para descarga
    """
    
    if len(top_short_df) == 0:
        return "No hay reglas SHORT disponibles para descargar"
    
    # Crear DataFrame para descarga con información relevante
    download_df = top_short_df.copy()
    
    # Seleccionar y renombrar columnas
    columns_download = {
        'descripcion': 'Variables',
        'tipo': 'Tipo',
        'regla_compuesta': 'Regla_Compuesta',
        'rendimiento_promedio': 'Rendimiento_Promedio',
        'std_rendimientos': 'Desviacion_Estandar',
        'ratio_estabilidad': 'Ratio_Estabilidad',
        'rendimiento_total_original': 'Rendimiento_Total_Original',
        'operaciones_total_original': 'Operaciones_Total'
    }
    
    # Filtrar solo las columnas que existen
    available_columns = {k: v for k, v in columns_download.items() if k in download_df.columns}
    
    download_df = download_df[list(available_columns.keys())].copy()
    download_df = download_df.rename(columns=available_columns)
    
    # Redondear valores numéricos para mejor legibilidad
    numeric_columns = download_df.select_dtypes(include=[np.number]).columns
    download_df[numeric_columns] = download_df[numeric_columns].round(6)
    
    # Convertir a CSV
    csv_string = download_df.to_csv(index=False, encoding='utf-8')
    
    return csv_string

def generar_resumen_reglas_ensemble(top_long_df, top_short_df, config_ensemble):
    """
    Genera un resumen completo del análisis ENSEMBLE para descarga
    
    Parameters:
    -----------
    top_long_df : pandas.DataFrame
        DataFrame con reglas LONG
    top_short_df : pandas.DataFrame  
        DataFrame con reglas SHORT
    config_ensemble : dict
        Configuración usada en el ENSEMBLE
        
    Returns:
    --------
    str
        Resumen en formato texto para descarga
    """
    
    resumen = []
    
    # Header
    resumen.append("=" * 80)
    resumen.append("RESUMEN DEL ANÁLISIS ENSEMBLE")
    resumen.append("=" * 80)
    resumen.append("")
    
    # Configuración
    resumen.append("CONFIGURACIÓN:")
    resumen.append(f"• Número de trozos: {config_ensemble.get('n_trozos', 'N/A')}")
    resumen.append(f"• Reglas por tipo seleccionadas: {config_ensemble.get('n_reglas', 'N/A')}")
    resumen.append("")
    
    # Estadísticas generales
    total_reglas = len(top_long_df) + len(top_short_df)
    resumen.append("ESTADÍSTICAS GENERALES:")
    resumen.append(f"• Total reglas seleccionadas: {total_reglas}")
    resumen.append(f"• Reglas LONG: {len(top_long_df)}")
    resumen.append(f"• Reglas SHORT: {len(top_short_df)}")
    resumen.append("")
    
    # Top reglas LONG
    if len(top_long_df) > 0:
        resumen.append("TOP REGLAS LONG:")
        resumen.append("-" * 40)
        for i, (_, regla) in enumerate(top_long_df.head(5).iterrows(), 1):
            resumen.append(f"{i}. {regla['descripcion']}")
            resumen.append(f"   Ratio Estabilidad: {regla['ratio_estabilidad']:.3f}")
            resumen.append(f"   Rendimiento Promedio: {regla['rendimiento_promedio']:.6f}")
            resumen.append(f"   Desv. Estándar: {regla['std_rendimientos']:.6f}")
            resumen.append("")
    
    # Top reglas SHORT
    if len(top_short_df) > 0:
        resumen.append("TOP REGLAS SHORT:")
        resumen.append("-" * 40)
        for i, (_, regla) in enumerate(top_short_df.head(5).iterrows(), 1):
            resumen.append(f"{i}. {regla['descripcion']}")
            resumen.append(f"   Ratio Estabilidad: {regla['ratio_estabilidad']:.3f}")
            resumen.append(f"   Rendimiento Promedio: {regla['rendimiento_promedio']:.6f}")
            resumen.append(f"   Desv. Estándar: {regla['std_rendimientos']:.6f}")
            resumen.append("")
    
    # Interpretación
    resumen.append("INTERPRETACIÓN:")
    resumen.append("• Ratio de Estabilidad = Rendimiento Promedio / Desviación Estándar")
    resumen.append("• Ratios altos indican rendimientos consistentes y predecibles")
    resumen.append("• Ratios bajos indican rendimientos variables e impredecibles")
    resumen.append("• Las mejores reglas combinan buen rendimiento con baja variabilidad")
    resumen.append("")
    
    # Footer
    resumen.append("=" * 80)
    resumen.append(f"Reporte generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    resumen.append("=" * 80)
    
    return "\n".join(resumen)

def crear_grafico_equity_individual_simple(equity_data):
    """
    Crea gráfico con equity curves individuales SIMPLIFICADO
    Solo muestra colores por tipo (LONG=verde, SHORT=rojo) sin leyenda detallada
    
    Parameters:
    -----------
    equity_data : dict
        Diccionario resultado de generar_equity_curves
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Figura de Plotly con equity curves simplificadas
    """
    
    fig = go.Figure()
    
    # Contadores para leyenda simplificada
    long_added = False
    short_added = False
    benchmark_added = False
    
    # Añadir cada equity curve
    for regla_name, data in equity_data.items():
        
        # Determinar color y estilo según tipo
        if data['tipo'] == 'LONG':
            color = '#2E8B57'  # Verde
            show_legend = not long_added
            legend_name = 'Reglas LONG'
            long_added = True
        elif data['tipo'] == 'SHORT':
            color = '#DC143C'  # Rojo
            show_legend = not short_added
            legend_name = 'Reglas SHORT'
            short_added = True
        else:  # BENCHMARK
            color = '#808080'  # Gris
            show_legend = not benchmark_added
            legend_name = 'Benchmark (Buy & Hold)'
            benchmark_added = True
        
        # Estilo de línea
        if data['tipo'] == 'BENCHMARK':
            line_dash = 'dash'
            line_width = 2
        else:
            line_dash = 'solid'
            line_width = 1.5
        
        fig.add_trace(go.Scatter(
            x=data['fechas'],
            y=data['equity'],
            mode='lines',
            name=legend_name,
            legendgroup=data['tipo'],  # Agrupar por tipo
            showlegend=show_legend,    # Solo mostrar una vez por tipo
            line=dict(color=color, dash=line_dash, width=line_width),
            hovertemplate=(
                f'<b>{data["descripcion"]}</b><br>' +
                'Fecha: %{x}<br>' +
                'Equity: %{y:.6f}<br>' +
                f'Tipo: {data["tipo"]}<br>' +
                f'Operaciones: {data["operaciones"]}<br>' +
                f'Rendimiento Total: {data["rendimiento_total"]:.6f}<extra></extra>'
            )
        ))
    
    # Layout
    fig.update_layout(
        title=dict(
            text='📈 Equity Curves por Tipo de Regla',
            x=0.5,
            font=dict(size=20)
        ),
        xaxis_title='Fecha',
        yaxis_title='Equity Acumulada',
        hovermode='x unified',
        template='plotly_white',
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        )
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

def debug_ensemble_structure(top_long, top_short):
    """
    Función temporal para debuggear la estructura de las reglas del ENSEMBLE
    """
    print("🔍 DEBUG - Estructura del ENSEMBLE:")
    
    print(f"\n📈 TOP_LONG:")
    print(f"   Tipo: {type(top_long)}")
    print(f"   Filas: {len(top_long)}")
    if len(top_long) > 0:
        print(f"   Columnas: {list(top_long.columns)}")
        print(f"   Primer registro:")
        first_long = top_long.iloc[0]
        for col in top_long.columns:
            print(f"      {col}: {first_long[col]}")
    
    print(f"\n📉 TOP_SHORT:")
    print(f"   Tipo: {type(top_short)}")
    print(f"   Filas: {len(top_short)}")
    if len(top_short) > 0:
        print(f"   Columnas: {list(top_short.columns)}")
        print(f"   Primer registro:")
        first_short = top_short.iloc[0]
        for col in top_short.columns:
            print(f"      {col}: {first_short[col]}")

# ═══════════════════════════════════════════════════════════════
# FUNCIONES PARA ANÁLISIS DE OPTIMIZACIÓN MT5
# ═══════════════════════════════════════════════════════════════

def analizar_archivo_optimizacion_mt5(archivo_xml):
    """
    Analiza un archivo XML de resultados de optimización MT5 y calcula rangos
    recomendados para los parámetros.
    
    Parameters:
    -----------
    archivo_xml : UploadedFile
        Archivo XML subido desde Streamlit
        
    Returns:
    --------
    dict
        Diccionario con resultados del análisis
    """
    try:
        # Leer contenido del archivo
        contenido = archivo_xml.read()
        
        # Parsear XML desde string
        root = ET.fromstring(contenido)
        
        # Obtener namespace
        namespaces = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}
        
        # Buscar filas
        rows = root.findall('.//ss:Table/ss:Row', namespaces)
        
        if not rows:
            # Probar sin namespace
            rows = root.findall('.//Table/Row')
            
        if not rows or len(rows) <= 1:
            return {
                'success': False,
                'error': 'No se encontraron datos suficientes en el archivo XML'
            }
        
        # Extraer headers y datos
        header_row = rows[0]
        data_rows = rows[1:]
        
        # Extraer encabezados
        headers = []
        for cell in header_row.findall('.//ss:Cell/ss:Data', namespaces) or header_row.findall('.//Cell/Data'):
            if cell.text:
                headers.append(cell.text)
        
        # Extraer datos
        data = []
        for row in data_rows:
            row_data = []
            for cell in row.findall('.//ss:Cell/ss:Data', namespaces) or row.findall('.//Cell/Data'):
                value = cell.text if cell.text else ''
                
                # Intentar convertir a número
                if value:
                    try:
                        if '.' in str(value):
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass
                row_data.append(value)
            
            # Solo agregar filas completas
            if len(row_data) == len(headers):
                data.append(row_data)
        
        # Crear DataFrame
        df = pd.DataFrame(data, columns=headers)
        
        # Convertir columnas numéricas
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass
        
        # Verificar que existe columna Result
        if 'Result' not in df.columns:
            return {
                'success': False,
                'error': 'No se encontró la columna "Result" en el archivo'
            }
        
        return {
            'success': True,
            'dataframe': df,
            'total_filas': len(df),
            'columnas': list(df.columns)
        }
        
    except ET.ParseError as e:
        return {
            'success': False,
            'error': f'Error al parsear XML: {str(e)}'
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error procesando archivo: {str(e)}'
        }

def calcular_rangos_parametros_optimizacion(df):
    """
    Calcula rangos recomendados para parámetros de optimización
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame con resultados de optimización
        
    Returns:
    --------
    dict
        Diccionario con análisis y rangos calculados
    """
    
    if 'Result' not in df.columns:
        return {
            'success': False,
            'error': 'No se encontró la columna "Result"'
        }
    
    # Paso 1: Filtrar por Result > 0
    df_positivos = df[df['Result'] > 0].copy()
    
    if len(df_positivos) == 0:
        return {
            'success': False,
            'error': 'No hay resultados con Result > 0'
        }
    
    # Paso 2: Calcular promedio y filtrar por >= promedio
    result_promedio = df_positivos['Result'].mean()
    df_filtrado = df_positivos[df_positivos['Result'] >= result_promedio].copy()
    
    if len(df_filtrado) == 0:
        return {
            'success': False,
            'error': 'No hay resultados con Result >= promedio'
        }
    
    # Paso 3: Identificar columnas de parámetros (eliminar estándar)
    columnas_estandar = [
        'Pass', 'Result', 'Profit', 'Expected Payoff', 'Profit Factor', 
        'Recovery Factor', 'Sharpe Ratio', 'Custom', 'Equity DD %', 'Trades',
        'Balance DD %', 'Maximal Drawdown', 'Total Trades', 'Gross Profit',
        'Gross Loss', 'Profit Trades', 'Loss Trades', 'Largest Profit',
        'Largest Loss', 'Average Profit', 'Average Loss', 'Maximum Wins',
        'Maximum Losses', 'Average Win', 'Average Loss'
    ]
    
    # Filtrar solo columnas que existen
    columnas_a_eliminar = [col for col in columnas_estandar if col in df_filtrado.columns]
    df_parametros = df_filtrado.drop(columns=columnas_a_eliminar)
    
    # Paso 4: Calcular rangos para cada parámetro
    rangos_parametros = {}
    
    for columna in df_parametros.columns:
        media = df_parametros[columna].mean()
        std = df_parametros[columna].std()
        
        # Calcular rango: media ± 1 desviación estándar
        min_val = media - std
        max_val = media + std
        
        # Determinar si es parámetro de período (entero) o decimal
        if 'Period' in columna or 'period' in columna.lower():
            # Parámetros de período: enteros
            min_val = max(1, int(round(min_val)))  # Mínimo 1
            max_val = int(round(max_val))
            tipo = 'entero'
        else:
            # Otros parámetros: decimales
            min_val = round(min_val, 2)
            max_val = round(max_val, 2)
            tipo = 'decimal'
        
        rangos_parametros[columna] = {
            'min': min_val,
            'max': max_val,
            'media': round(media, 4),
            'std': round(std, 4),
            'tipo': tipo,
            'valores_originales': df_parametros[columna].tolist()
        }
    
    return {
        'success': True,
        'total_filas_original': len(df),
        'filas_result_positivo': len(df_positivos),
        'result_promedio': round(result_promedio, 2),
        'filas_finales': len(df_filtrado),
        'columnas_eliminadas': columnas_a_eliminar,
        'rangos_parametros': rangos_parametros,
        'df_filtrado': df_filtrado,
        'mejores_resultados': df_filtrado.nlargest(5, 'Result')[['Result'] + list(df_parametros.columns)]
    }

def generar_resumen_optimizacion(analisis_resultado):
    """
    Genera un resumen textual del análisis de optimización
    
    Parameters:
    -----------
    analisis_resultado : dict
        Resultado de calcular_rangos_parametros_optimizacion
        
    Returns:
    --------
    str
        Resumen en formato texto
    """
    
    if not analisis_resultado['success']:
        return f"Error en el análisis: {analisis_resultado['error']}"
    
    resumen = []
    
    # Header
    resumen.append("=" * 70)
    resumen.append("ANÁLISIS DE OPTIMIZACIÓN MT5 - RANGOS RECOMENDADOS")
    resumen.append("=" * 70)
    resumen.append("")
    
    # Estadísticas del filtrado
    resumen.append("ESTADÍSTICAS DEL PROCESO:")
    resumen.append(f"• Total de pruebas: {analisis_resultado['total_filas_original']:,}")
    resumen.append(f"• Pruebas con Result > 0: {analisis_resultado['filas_result_positivo']:,}")
    resumen.append(f"• Promedio de Result: {analisis_resultado['result_promedio']}")
    resumen.append(f"• Pruebas >= promedio: {analisis_resultado['filas_finales']:,}")
    resumen.append(f"• Tasa de éxito: {analisis_resultado['filas_finales']/analisis_resultado['total_filas_original']*100:.1f}%")
    resumen.append("")
    
    # Rangos de parámetros
    resumen.append("RANGOS RECOMENDADOS PARA PARÁMETROS:")
    resumen.append("-" * 50)
    
    for parametro, datos in analisis_resultado['rangos_parametros'].items():
        if datos['tipo'] == 'entero':
            resumen.append(f"{parametro}: {datos['min']} - {datos['max']} (media: {datos['media']:.0f})")
        else:
            resumen.append(f"{parametro}: {datos['min']} - {datos['max']} (media: {datos['media']:.2f})")
    
    resumen.append("")
    
    # Metodología
    resumen.append("METODOLOGÍA:")
    resumen.append("1. Filtrar solo resultados con Result > 0")
    resumen.append("2. Calcular promedio de Result y filtrar >= promedio")
    resumen.append("3. Calcular rangos usando: Media ± 1 Desviación Estándar")
    resumen.append("4. Parámetros 'Period': valores enteros")
    resumen.append("5. Otros parámetros: valores decimales (2 decimales)")
    resumen.append("")
    
    # Footer
    resumen.append("=" * 70)
    resumen.append(f"Análisis generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    resumen.append("=" * 70)
    
    return "\n".join(resumen)

def crear_grafico_distribucion_parametros(analisis_resultado):
    """
    Crea gráficos de distribución para los parámetros optimizados
    
    Parameters:
    -----------
    analisis_resultado : dict
        Resultado de calcular_rangos_parametros_optimizacion
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Figura con histogramas de parámetros
    """
    
    if not analisis_resultado['success']:
        return None
    
    rangos = analisis_resultado['rangos_parametros']
    
    if len(rangos) == 0:
        return None
    
    # Calcular número de subplots
    n_params = len(rangos)
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    # Crear subplots (sin títulos, los agregaremos como anotaciones)
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=list(rangos.keys()),
        vertical_spacing=0.4,
        horizontal_spacing=0.1
    )

    # Añadir histograma para cada parámetro
    for i, (param_name, param_data) in enumerate(rangos.items()):
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1

        # Datos del parámetro
        valores = param_data['valores_originales']

        # Añadir histograma
        fig.add_trace(
            go.Histogram(
                x=valores,
                name=param_name,
                showlegend=False,
                marker_color='skyblue',
                opacity=0.7,
                nbinsx=min(20, len(set(valores)))
            ),
            row=row, col=col
        )


        # Añadir líneas para rango recomendado
        fig.add_vline(
            x=param_data['min'],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Min: {param_data['min']}",
            annotation_position="top",
            row=row, col=col
        )

        fig.add_vline(
            x=param_data['max'],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Max: {param_data['max']}",
            annotation_position="top",
            row=row, col=col
        )

        # Línea de media
        fig.add_vline(
            x=param_data['media'],
            line_dash="solid",
            line_color="green",
            annotation_text=f"Media: {param_data['media']:.2f}",
            annotation_position="bottom",
            row=row, col=col
        )
    
    # Ajustar posición vertical de los títulos (anotaciones de los subplots)
    for annotation in fig['layout']['annotations']:
        annotation['y'] += 0.08  # Sube un poco los títulos
        # Opcional: también puedes reducir el tamaño o cambiar estilo si es necesario
        # annotation['font'] = dict(size=12)

    # Layout
    fig.update_layout(
        title=dict(
            text='📊 Distribución de Parámetros Optimizados',
            x=0.5,
            font=dict(size=16)
        ),
        height=200 * n_rows + 100,
        template='plotly_white',
        showlegend=False
    )
    
    return fig

# ═══════════════════════════════════════════════════════════════
# FUNCIONES PARA ANÁLISIS DE POSITION SIZING (TAB SIZING)
# ═══════════════════════════════════════════════════════════════

def analizar_reporte_mt5_html(archivo_html):
    """
    Analiza un archivo HTML de reporte de backtest de MT5 y extrae las operaciones
    Basado en el script original simplificado
    
    Parameters:
    -----------
    archivo_html : UploadedFile
        Archivo HTML subido desde Streamlit (reporte de MT5)
        
    Returns:
    --------
    dict
        Diccionario con información extraída del reporte
    """
    
    try:
        # Reiniciar puntero del archivo
        archivo_html.seek(0)
        
        # Guardar archivo temporalmente para que pandas.read_html pueda leerlo
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.html', delete=False) as temp_file:
            # Escribir contenido del archivo subido al archivo temporal
            temp_file.write(archivo_html.read())
            temp_file_path = temp_file.name
        
        try:
            # EXACTAMENTE como en el script original
            tablas = pd.read_html(temp_file_path)
            data = tablas[1]  # Segunda tabla como en el original
            
            print(f"🔍 DEBUG - Tabla extraída:")
            print(f"   Filas: {data.shape[0]}, Columnas: {data.shape[1]}")
            print(f"   Columnas disponibles: {list(data.columns)}")
            
            # Verificar que tiene las columnas necesarias
            if data.shape[1] < 11:
                return {
                    'success': False,
                    'error': f'La tabla debe tener al menos 11 columnas, encontradas: {data.shape[1]}. Estructura: {list(data.columns)}'
                }
            
            # EXACTAMENTE como en el script original - procesar columnas 9 y 10
            print(f"   Procesando columna 10 (antes): {data[10].dtype}")
            data[10] = data[10].astype(str).str.replace(' ', '')  # Eliminar espacios
            data[10] = data[10].astype(str).str.replace(',', '.')  # Reemplazar comas por puntos
            data[10] = pd.to_numeric(data[10], errors='coerce')   # Convertir a numérico
            
            print(f"   Procesando columna 9 (antes): {data[9].dtype}")
            data[9] = data[9].astype(str).str.replace(' ', '')   # Eliminar espacios
            data[9] = data[9].astype(str).str.replace(',', '.')  # Reemplazar comas por puntos
            data[9] = pd.to_numeric(data[9], errors='coerce')    # Convertir a numérico
            
            # EXACTAMENTE como en el script original - filtrar por 'out'
            print(f"   Filtrando por columna 4 = 'out'")
            print(f"   Valores únicos en columna 4: {data[4].unique()}")
            data = data[(data[4] == 'out')]
            
            if len(data) == 0:
                return {
                    'success': False,
                    'error': 'No se encontraron operaciones con estado "out". Valores encontrados en columna 4: ' + str(list(data[4].unique()))
                }
            
            # EXACTAMENTE como en el script original - calcular retornos
            data['Retornos'] = data[9] + data[10]
            trades = data['Retornos']
            
            # Limpiar NaN
            trades_clean = trades.dropna()
            
            if len(trades_clean) == 0:
                return {
                    'success': False,
                    'error': 'No se encontraron retornos válidos después del procesamiento'
                }
            
            print(f"   ✅ Operaciones extraídas: {len(trades_clean)}")
            print(f"   Suma total: {trades_clean.sum():.2f}")
            print(f"   Promedio: {trades_clean.mean():.6f}")
            
            # Extraer fechas del PERÍODO del backtest (del encabezado HTML)
            fecha_inicio = None
            fecha_fin = None
            
            try:
                # Leer el contenido HTML completo para buscar las fechas del período
                archivo_html.seek(0)
                contenido_html = archivo_html.read()
                
                if isinstance(contenido_html, bytes):
                    contenido_html = contenido_html.decode('utf-8', errors='ignore')
                
                print(f"   🔍 Buscando fechas del período en el HTML...")
                print(f"   📄 Tamaño total del HTML: {len(contenido_html)} caracteres")
                
                # Buscar patrones típicos de MT5 para el período del backtest
                import re
                
                # PATRONES CORREGIDOS basados en el HTML real
                patrones_periodo = [
                    # Patrón exacto: "Período: Daily (2006.06.02 - 2024.12.31)"
                    r'Período:\s*Daily\s*\((\d{4}\.\d{2}\.\d{2})\s*-\s*(\d{4}\.\d{2}\.\d{2})\)',
                    r'Period:\s*Daily\s*\((\d{4}\.\d{2}\.\d{2})\s*-\s*(\d{4}\.\d{2}\.\d{2})\)',
                    r'Periodo:\s*Daily\s*\((\d{4}\.\d{2}\.\d{2})\s*-\s*(\d{4}\.\d{2}\.\d{2})\)',
                    
                    # Variaciones sin los dos puntos
                    r'Período\s*Daily\s*\((\d{4}\.\d{2}\.\d{2})\s*-\s*(\d{4}\.\d{2}\.\d{2})\)',
                    r'Period\s*Daily\s*\((\d{4}\.\d{2}\.\d{2})\s*-\s*(\d{4}\.\d{2}\.\d{2})\)',
                    r'Periodo\s*Daily\s*\((\d{4}\.\d{2}\.\d{2})\s*-\s*(\d{4}\.\d{2}\.\d{2})\)',
                    
                    # Patrón más general para Daily
                    r'Daily\s*\((\d{4}\.\d{2}\.\d{2})\s*-\s*(\d{4}\.\d{2}\.\d{2})\)',
                    
                    # Patrón muy general para cualquier par de fechas con guión
                    r'(\d{4}\.\d{2}\.\d{2})\s*-\s*(\d{4}\.\d{2}\.\d{2})'
                ]
                
                for i, patron in enumerate(patrones_periodo):
                    print(f"   🔍 Probando patrón {i+1}: {patron}")
                    match = re.search(patron, contenido_html, re.IGNORECASE)
                    if match:
                        fecha_inicio_str = match.group(1)
                        fecha_fin_str = match.group(2)
                        print(f"   ✅ Match encontrado! {fecha_inicio_str} → {fecha_fin_str}")
                        
                        try:
                            # Formato típico de MT5: YYYY.MM.DD
                            fecha_inicio = pd.to_datetime(fecha_inicio_str, format='%Y.%m.%d')
                            fecha_fin = pd.to_datetime(fecha_fin_str, format='%Y.%m.%d')
                            
                            duracion = (fecha_fin - fecha_inicio).days
                            print(f"   📊 Duración: {duracion} días ({duracion/365.25:.2f} años)")
                            
                            # Verificar que las fechas sean razonables
                            if duracion > 30:  # Al menos 30 días
                                print(f"   ✅ Fechas del período extraídas: {fecha_inicio.date()} → {fecha_fin.date()}")
                                print(f"   📊 Patrón usado: {patron}")
                                break
                            else:
                                print(f"   ⚠️ Duración muy corta ({duracion} días), buscando otras fechas...")
                                
                        except Exception as e:
                            print(f"   ⚠️ Error parseando fechas {fecha_inicio_str}, {fecha_fin_str}: {str(e)}")
                            continue
                    else:
                        print(f"   ❌ No match para patrón {i+1}")
                
                # Si no se encontraron con los patrones, buscar en contexto específico
                if fecha_inicio is None:
                    print(f"   🔍 Búsqueda de contexto específico...")
                    
                    # Buscar texto alrededor de "Período" o "Daily"
                    contextos = []
                    for keyword in ['Período', 'Period', 'Periodo', 'Daily']:
                        pos = contenido_html.find(keyword)
                        if pos != -1:
                            contexto = contenido_html[max(0, pos-50):pos+200]
                            contextos.append(contexto)
                            print(f"   📄 Contexto '{keyword}': {contexto}")
                    
                    # Buscar fechas en los contextos
                    for contexto in contextos:
                        fechas_contexto = re.findall(r'\d{4}\.\d{2}\.\d{2}', contexto)
                        if len(fechas_contexto) >= 2:
                            try:
                                fecha_inicio = pd.to_datetime(fechas_contexto[0], format='%Y.%m.%d')
                                fecha_fin = pd.to_datetime(fechas_contexto[1], format='%Y.%m.%d')
                                
                                # Asegurar orden correcto
                                if fecha_fin < fecha_inicio:
                                    fecha_inicio, fecha_fin = fecha_fin, fecha_inicio
                                
                                duracion = (fecha_fin - fecha_inicio).days
                                print(f"   ✅ Fechas extraídas del contexto: {fecha_inicio.date()} → {fecha_fin.date()}")
                                print(f"   📊 Duración: {duracion} días ({duracion/365.25:.2f} años)")
                                break
                            except Exception as e:
                                print(f"   ⚠️ Error en contexto: {str(e)}")
                                continue
                
                # Si no se encontraron fechas del período, intentar extraer de la tabla como fallback
                if fecha_inicio is None:
                    print(f"   🔄 Fallback: Extrayendo fechas de la tabla de operaciones...")
                    
                    fechas_col = tablas[1][0].astype(str)  # Primera columna de la tabla original
                    fechas_validas = []
                    
                    for fecha_str in fechas_col.head(20):  # Probar las primeras 20
                        if pd.isna(fecha_str) or fecha_str in ['nan', '', 'None']:
                            continue
                        
                        # Formatos típicos de MT5
                        formatos = [
                            '%Y.%m.%d %H:%M:%S',
                            '%Y.%m.%d %H:%M',
                            '%d.%m.%Y %H:%M:%S',
                            '%d.%m.%Y %H:%M',
                            '%Y-%m-%d %H:%M:%S',
                            '%Y-%m-%d %H:%M'
                        ]
                        
                        for formato in formatos:
                            try:
                                fecha_parseada = pd.to_datetime(fecha_str, format=formato)
                                fechas_validas.append(fecha_parseada)
                                break
                            except:
                                continue
                    
                    if len(fechas_validas) >= 2:
                        fecha_inicio = min(fechas_validas)
                        fecha_fin = max(fechas_validas)
                        print(f"   📅 Fechas de tabla extraídas: {fecha_inicio.date()} → {fecha_fin.date()}")
                
            except Exception as e:
                print(f"   ⚠️ Error extrayendo fechas: {str(e)}")
            
            return {
                'success': True,
                'retornos': trades_clean.tolist(),
                'num_operaciones': len(trades_clean),
                'fecha_inicio': fecha_inicio,
                'fecha_fin': fecha_fin,
                'suma_total': trades_clean.sum(),
                'retorno_promedio': trades_clean.mean(),
                'retorno_std': trades_clean.std(),
                'data_original': data
            }
            
        finally:
            # Limpiar archivo temporal
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ Error completo: {error_detail}")
        
        return {
            'success': False,
            'error': f'Error procesando archivo: {str(e)}'
        }

def calcular_operaciones_por_año(fecha_inicio, fecha_fin, num_operaciones):
    """
    Calcula el ratio de operaciones por año basado en las fechas del backtest
    
    Parameters:
    -----------
    fecha_inicio : datetime
        Fecha de inicio del backtest
    fecha_fin : datetime  
        Fecha de fin del backtest
    num_operaciones : int
        Número total de operaciones
        
    Returns:
    --------
    dict
        Información sobre operaciones por año
    """
    
    try:
        # Calcular duración en días
        duracion_dias = (fecha_fin - fecha_inicio).days
        
        if duracion_dias <= 0:
            return {
                'success': False,
                'error': 'La duración del backtest debe ser positiva'
            }
        
        # Calcular duración en años
        duracion_años = duracion_dias / 365.25  # Considerar años bisiestos
        
        # Calcular operaciones por año
        operaciones_por_año = num_operaciones / duracion_años
        
        return {
            'success': True,
            'duracion_dias': duracion_dias,
            'duracion_años': round(duracion_años, 2),
            'operaciones_por_año': round(operaciones_por_año, 1),
            'operaciones_por_año_entero': int(round(operaciones_por_año))
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Error calculando operaciones por año: {str(e)}'
        }

def calcular_maxdd(retornos, capital_inicial=1000):
    """
    Calcula el drawdown máximo absoluto y relativo de una serie de retornos
    
    Parameters:
    -----------
    retornos : list or array
        Lista de retornos de operaciones
    capital_inicial : float
        Capital inicial para el cálculo
        
    Returns:
    --------
    dict
        Diccionario con métricas de drawdown
    """
    
    try:
        retornos_array = np.array(retornos)
        
        # Paso 1: Calcular retornos acumulados (sin reinversión)
        retornos_acumulados = np.cumsum(retornos_array) + capital_inicial
        
        # Paso 2: Calcular el pico móvil (máximo histórico hasta cada punto)
        pico_movil = np.maximum.accumulate(retornos_acumulados)
        
        # Paso 3: Calcular el drawdown en cada punto (valor absoluto)
        drawdowns_abs = pico_movil - retornos_acumulados
        
        # Paso 4: Calcular el drawdown en cada punto (valor relativo al pico)
        drawdowns_rel = np.where(pico_movil > 0, drawdowns_abs / pico_movil * 100, 0)
        
        # Paso 5: Encontrar el drawdown máximo (absoluto y relativo)
        max_dd_abs = np.max(drawdowns_abs)
        max_dd_rel = np.max(drawdowns_rel)
        
        # Puntos para visualización
        idx_max = np.argmax(drawdowns_abs) if len(drawdowns_abs) > 0 else 0
        pico_idx = np.where(retornos_acumulados[:idx_max] == pico_movil[idx_max])[0][-1] if idx_max > 0 else 0
        
        return {
            'max_dd_abs': max_dd_abs,
            'max_dd_rel': max_dd_rel,
            'pico_idx': pico_idx,
            'idx_max': idx_max,
            'retornos_acumulados': retornos_acumulados,
            'pico_movil': pico_movil,
            'drawdowns_abs': drawdowns_abs,
            'drawdowns_rel': drawdowns_rel,
            'capital_inicial': capital_inicial,
            'capital_final': retornos_acumulados[-1] if len(retornos_acumulados) > 0 else capital_inicial
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Error calculando drawdown: {str(e)}'
        }

def simulacion_monte_carlo_sizing(retornos_historicos, operaciones_por_año, n_simulaciones=10000, capital_inicial=2000):
    """
    Ejecuta simulaciones Monte Carlo para estimar distribución de drawdown máximo
    
    Parameters:
    -----------
    retornos_historicos : list
        Lista de retornos históricos de operaciones
    operaciones_por_año : int
        Número de operaciones a simular por año
    n_simulaciones : int
        Número de simulaciones a ejecutar
    capital_inicial : float
        Capital inicial para cada simulación
        
    Returns:
    --------
    dict
        Resultados de las simulaciones
    """
    
    import random
    
    try:
        if len(retornos_historicos) < operaciones_por_año:
            return {
                'success': False,
                'error': f'No hay suficientes operaciones históricas ({len(retornos_historicos)}) para simular {operaciones_por_año} operaciones por año'
            }
        
        print(f"🔄 Ejecutando {n_simulaciones:,} simulaciones Monte Carlo...")
        print(f"   Operaciones por simulación: {operaciones_por_año}")
        print(f"   Capital inicial: {capital_inicial}")
        
        max_dd_list = []
        profits_list = []
        
        for i in range(n_simulaciones):
            # Mostrar progreso cada 25%
            if n_simulaciones >= 4:
                quarter_points = [n_simulaciones // 4, n_simulaciones // 2, 
                                3 * n_simulaciones // 4, n_simulaciones]
                if (i + 1) in quarter_points:
                    porcentaje = int((i + 1) / n_simulaciones * 100)
                    print(f"   Progreso: {porcentaje}% ({i + 1:,}/{n_simulaciones:,} simulaciones)")
            
            # Generar muestra aleatoria
            muestra = random.sample(retornos_historicos, operaciones_por_año)
            
            # Calcular drawdown máximo
            resultado_dd = calcular_maxdd(muestra, capital_inicial)
            max_dd = round(resultado_dd['max_dd_abs'], 2)
            
            # Calcular profit total
            profit = round(np.sum(muestra), 2)
            
            max_dd_list.append(max_dd)
            profits_list.append(profit)
        
        # Calcular estadísticas
        max_dd_array = np.array(max_dd_list)
        profits_array = np.array(profits_list)
        
        # Percentiles de drawdown
        percentiles = [50, 80, 90, 95, 99]
        percentiles_dd = {p: np.percentile(max_dd_array, p) for p in percentiles}
        
        # Percentiles de profit
        percentiles_profit = {p: np.percentile(profits_array, p) for p in percentiles}
        
        return {
            'success': True,
            'max_dd_list': max_dd_list,
            'profits_list': profits_list,
            'operaciones_por_año': operaciones_por_año,
            'n_simulaciones': n_simulaciones,
            'capital_inicial': capital_inicial,
            
            # Estadísticas de drawdown
            'dd_mean': np.mean(max_dd_array),
            'dd_median': np.median(max_dd_array),
            'dd_std': np.std(max_dd_array),
            'dd_min': np.min(max_dd_array),
            'dd_max': np.max(max_dd_array),
            'dd_percentiles': percentiles_dd,
            
            # Estadísticas de profit
            'profit_mean': np.mean(profits_array),
            'profit_median': np.median(profits_array),
            'profit_std': np.std(profits_array),
            'profit_min': np.min(profits_array),
            'profit_max': np.max(profits_array),
            'profit_percentiles': percentiles_profit,
            
            # Ratios
            'profit_dd_ratio_mean': np.mean(profits_array) / np.mean(max_dd_array) if np.mean(max_dd_array) > 0 else 0
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Error en simulación Monte Carlo: {str(e)}'
        }

def crear_grafico_drawdown_historico(resultado_dd):
    """
    Crea gráfico del drawdown histórico
    
    Parameters:
    -----------
    resultado_dd : dict
        Resultado de calcular_maxdd
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Gráfico del drawdown histórico
    """
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Evolución de Capital', 'Drawdown Absoluto'),
        vertical_spacing=0.1,
        shared_xaxes=True
    )
    
    # Gráfico 1: Evolución de capital
    x_vals = list(range(len(resultado_dd['retornos_acumulados'])))
    
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=resultado_dd['retornos_acumulados'],
            mode='lines',
            name='Capital Acumulado',
            line=dict(color='blue', width=2),
            hovertemplate='Operación: %{x}<br>Capital: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=resultado_dd['pico_movil'],
            mode='lines',
            name='Pico Móvil',
            line=dict(color='green', dash='dash', width=1),
            hovertemplate='Operación: %{x}<br>Pico: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Gráfico 2: Drawdown absoluto
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=resultado_dd['drawdowns_abs'],
            mode='lines',
            name='Drawdown Absoluto',
            line=dict(color='red', width=2),
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.1)',
            hovertemplate='Operación: %{x}<br>Drawdown: %{y:.2f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Marcar punto de máximo drawdown
    fig.add_trace(
        go.Scatter(
            x=[resultado_dd['idx_max']],
            y=[resultado_dd['max_dd_abs']],
            mode='markers',
            name=f'Max DD: {resultado_dd["max_dd_abs"]:.2f}',
            marker=dict(color='red', size=10, symbol='circle'),
            hovertemplate=f'Máximo DD: {resultado_dd["max_dd_abs"]:.2f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f'📉 Análisis de Drawdown Histórico (Max DD: {resultado_dd["max_dd_abs"]:.2f})',
            x=0.5,
            font=dict(size=16)
        ),
        height=600,
        template='plotly_white',
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Número de Operación", row=2, col=1)
    fig.update_yaxes(title_text="Capital", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", row=2, col=1)
    
    return fig

def crear_grafico_distribucion_monte_carlo(simulacion_resultado, metric='drawdown'):
    """
    Crea gráfico de distribución de las simulaciones Monte Carlo
    
    Parameters:
    -----------
    simulacion_resultado : dict
        Resultado de simulacion_monte_carlo_sizing
    metric : str
        'drawdown' o 'profit' - métrica a graficar
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Gráfico de distribución
    """
    
    if metric == 'drawdown':
        data = simulacion_resultado['max_dd_list']
        title = '📊 Distribución de Drawdown Máximo (Monte Carlo)'
        x_title = 'Drawdown Máximo'
        color = '#DC143C'
        percentiles = simulacion_resultado['dd_percentiles']
        mean_val = simulacion_resultado['dd_mean']
        median_val = simulacion_resultado['dd_median']
    else:  # profit
        data = simulacion_resultado['profits_list']
        title = '📊 Distribución de Profit Total (Monte Carlo)'
        x_title = 'Profit Total'
        color = '#2E8B57'
        percentiles = simulacion_resultado['profit_percentiles']
        mean_val = simulacion_resultado['profit_mean']
        median_val = simulacion_resultado['profit_median']
    
    fig = go.Figure()
    
    # Histograma principal
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=50,
        name=x_title,
        marker_color=color,
        opacity=0.7,
        hovertemplate=f'<b>{x_title}:</b> %{{x}}<br><b>Frecuencia:</b> %{{y}}<extra></extra>'
    ))
    
    # Líneas de percentiles importantes
    percentiles_importantes = [80, 90, 95, 99]
    colores_percentiles = ['orange', 'red', 'darkred', 'purple']
    
    for p, color_p in zip(percentiles_importantes, colores_percentiles):
        fig.add_vline(
            x=percentiles[p],
            line_dash="dash",
            line_color=color_p,
            line_width=2,
            annotation_text=f"P{p}: {percentiles[p]:.2f}",
            annotation_position="top"
        )
    
    # Línea de media
    fig.add_vline(
        x=mean_val,
        line_dash="solid",
        line_color="blue",
        line_width=2,
        annotation_text=f"Media: {mean_val:.2f}",
        annotation_position="bottom left"
    )
    
    # Layout
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis_title=x_title,
        yaxis_title='Frecuencia',
        template='plotly_white',
        height=500,
        showlegend=False,
        annotations=[
            dict(
                x=0.02, y=0.98,
                xref='paper', yref='paper',
                text=(f'<b>Estadísticas:</b><br>'
                     f'Simulaciones: {simulacion_resultado["n_simulaciones"]:,}<br>'
                     f'Ops/año: {simulacion_resultado["operaciones_por_año"]}<br>'
                     f'Capital inicial: {simulacion_resultado["capital_inicial"]}<br>'
                     f'Media: {mean_val:.2f}<br>'
                     f'Mediana: {median_val:.2f}<br>'
                     f'P95: {percentiles[95]:.2f}<br>'
                     f'P99: {percentiles[99]:.2f}'),
                showarrow=False,
                align='left',
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='gray',
                borderwidth=1,
                font=dict(size=10, family='monospace')
            )
        ]
    )
    
    return fig

def generar_reporte_sizing(analisis_mt5, simulacion_resultado):
    """
    Genera reporte completo del análisis de position sizing
    
    Parameters:
    -----------
    analisis_mt5 : dict
        Resultado del análisis del reporte MT5
    simulacion_resultado : dict
        Resultado de las simulaciones Monte Carlo
        
    Returns:
    --------
    str
        Reporte en formato texto
    """
    
    reporte = []
    
    # Header
    reporte.append("=" * 80)
    reporte.append("ANÁLISIS DE POSITION SIZING - REPORTE MONTE CARLO")
    reporte.append("=" * 80)
    reporte.append("")
    
    # Información del backtest
    reporte.append("INFORMACIÓN DEL BACKTEST:")
    reporte.append(f"• Total de operaciones: {analisis_mt5['num_operaciones']:,}")
    if analisis_mt5['fecha_inicio'] and analisis_mt5['fecha_fin']:
        reporte.append(f"• Período: {analisis_mt5['fecha_inicio'].strftime('%Y-%m-%d')} → {analisis_mt5['fecha_fin'].strftime('%Y-%m-%d')}")
    reporte.append(f"• Retorno total: {analisis_mt5['suma_total']:.2f}")
    reporte.append(f"• Retorno promedio por operación: {analisis_mt5['retorno_promedio']:.4f}")
    reporte.append(f"• Desviación estándar: {analisis_mt5['retorno_std']:.4f}")
    reporte.append("")
    
    # Configuración de simulación
    reporte.append("CONFIGURACIÓN DE SIMULACIÓN:")
    reporte.append(f"• Operaciones por año simuladas: {simulacion_resultado['operaciones_por_año']}")
    reporte.append(f"• Número de simulaciones: {simulacion_resultado['n_simulaciones']:,}")
    reporte.append(f"• Capital inicial: {simulacion_resultado['capital_inicial']}")
    reporte.append("")
    
    # Resultados de drawdown
    reporte.append("RESULTADOS - DRAWDOWN MÁXIMO:")
    reporte.append(f"• Media: {simulacion_resultado['dd_mean']:.2f}")
    reporte.append(f"• Mediana: {simulacion_resultado['dd_median']:.2f}")
    reporte.append(f"• Desviación estándar: {simulacion_resultado['dd_std']:.2f}")
    reporte.append(f"• Mínimo: {simulacion_resultado['dd_min']:.2f}")
    reporte.append(f"• Máximo: {simulacion_resultado['dd_max']:.2f}")
    reporte.append("")
    
    reporte.append("PERCENTILES DE DRAWDOWN MÁXIMO:")
    for p, val in simulacion_resultado['dd_percentiles'].items():
        reporte.append(f"• Percentil {p}%: {val:.2f}")
    reporte.append("")
    
    # Resultados de profit
    reporte.append("RESULTADOS - PROFIT ANUAL:")
    reporte.append(f"• Media: {simulacion_resultado['profit_mean']:.2f}")
    reporte.append(f"• Mediana: {simulacion_resultado['profit_median']:.2f}")
    reporte.append(f"• Desviación estándar: {simulacion_resultado['profit_std']:.2f}")
    reporte.append("")
    
    # Ratio profit/drawdown
    reporte.append("RATIO PROFIT/DRAWDOWN:")
    reporte.append(f"• Ratio promedio: {simulacion_resultado['profit_dd_ratio_mean']:.3f}")
    reporte.append("")
    
    # Recomendaciones
    reporte.append("RECOMENDACIONES DE POSITION SIZING:")
    p95_dd = simulacion_resultado['dd_percentiles'][95]
    p99_dd = simulacion_resultado['dd_percentiles'][99]
    
    reporte.append(f"• Para 95% de confianza: Preparar capital para DD máximo de {p95_dd:.2f}")
    reporte.append(f"• Para 99% de confianza: Preparar capital para DD máximo de {p99_dd:.2f}")
    reporte.append("")
    
    if simulacion_resultado['capital_inicial'] > 0:
        size_95 = simulacion_resultado['capital_inicial'] / p95_dd if p95_dd > 0 else float('inf')
        size_99 = simulacion_resultado['capital_inicial'] / p99_dd if p99_dd > 0 else float('inf')
        
        reporte.append("TAMAÑO DE POSICIÓN RECOMENDADO:")
        reporte.append(f"• Para 95% confianza: {size_95:.2f}% del capital por trade")
        reporte.append(f"• Para 99% confianza: {size_99:.2f}% del capital por trade")
        reporte.append("")
    
    # Metodología
    reporte.append("METODOLOGÍA:")
    reporte.append("1. Extracción de retornos históricos del reporte MT5")
    reporte.append("2. Cálculo de operaciones por año basado en duración del backtest")
    reporte.append("3. Simulaciones Monte Carlo con muestras aleatorias")
    reporte.append("4. Cálculo de drawdown máximo para cada simulación")
    reporte.append("5. Análisis estadístico de la distribución resultante")
    reporte.append("")
    
    # Footer
    reporte.append("=" * 80)
    reporte.append(f"Reporte generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    reporte.append("=" * 80)
    
    return "\n".join(reporte)