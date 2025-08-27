import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import plotly.express as px

def analisar_localizacao_cd():
    """
    Função principal que realiza a análise comparativa entre Recife e Salvador
    para a instalação de um novo Centro de Distribuição.
    """

    # --- 1. Coleta e Estruturação de Dados ---
    # Em um projeto real, estes dados seriam coletados via APIs, bancos de dados ou web scraping.
    # Para esta simulação, os dados são baseados em pesquisas de mercado e logística.

    # 1.1 Custo Imobiliário (Valor médio do aluguel de galpão por m²)
    custos_imobiliarios = {
        'Cidade': ['Recife', 'Salvador'],
        'Custo_m2_Aluguel': [18.50, 21.00]  # Valores estimados com base em anúncios
    }
    df_custos = pd.DataFrame(custos_imobiliarios)

    # 1.2 Dados Logísticos (Distâncias em KM para as capitais do Nordeste)
    # Fonte: Pesquisas de rotas rodoviárias.
    capitais_ne = ['Aracaju', 'Maceió', 'João Pessoa', 'Natal', 'Fortaleza', 'Teresina', 'São Luís']
    distancias = {
        'Cidade': ['Recife', 'Salvador'],
        'Aracaju': [503, 321],
        'Maceió': [257, 580],
        'João Pessoa': [120, 933],
        'Natal': [290, 1101],
        'Fortaleza': [754, 1200],
        'Teresina': [1138, 1145],
        'São Luís': [1586, 1577]
    }
    df_distancias = pd.DataFrame(distancias)

    # 1.3 Dados Demográficos (PIB em Bilhões e População em Milhões dos estados vizinhos)
    # Fonte: Dados do IBGE e estudos econômicos regionais.
    # O potencial de consumo é uma métrica simplificada baseada na soma do PIB dos estados mais próximos.
    # Para Recife: PE, PB, RN, CE, AL. Para Salvador: BA, SE, AL, PE.
    pib_estados = {'BA': 29.0, 'PE': 17.7, 'CE': 15.4, 'MA': 10.1, 'RN': 6.8, 'PB': 6.2, 'AL': 5.5, 'PI': 5.2, 'SE': 4.1} # % do PIB do Nordeste
    potencial_consumo = {
        'Cidade': ['Recife', 'Salvador'],
        'Potencial_PIB_Vizinhos': [pib_estados['PE'] + pib_estados['PB'] + pib_estados['RN'] + pib_estados['CE'] + pib_estados['AL'],
                                   pib_estados['BA'] + pib_estados['SE'] + pib_estados['AL'] + pib_estados['PE']]
    }
    df_consumo = pd.DataFrame(potencial_consumo)

    # --- 2. Análise e Modelagem (Machine Learning Simples) ---

    # 2.1 Cálculo do Tempo Médio de Entrega
    # Assumindo velocidade média de um caminhão de 60 km/h.
    df_tempos = df_distancias.copy()
    for capital in capitais_ne:
        df_tempos[capital] = df_tempos[capital] / 60  # Converte distância para horas
    df_tempos['Tempo_Medio_Entrega'] = df_tempos[capitais_ne].mean(axis=1)

    # 2.2 Unificação dos Dados
    df_analise = pd.merge(df_custos, df_tempos[['Cidade', 'Tempo_Medio_Entrega']], on='Cidade')
    df_analise = pd.merge(df_analise, df_consumo, on='Cidade')

    # 2.3 Normalização dos Dados
    # Invertemos os custos e tempos, pois valores menores são melhores.
    df_analise['Inverso_Custo'] = 1 / df_analise['Custo_m2_Aluguel']
    df_analise['Inverso_Tempo_Medio'] = 1 / df_analise['Tempo_Medio_Entrega']

    scaler = MinMaxScaler()
    df_analise[['Score_Custo', 'Score_Logistica', 'Score_Consumo']] = scaler.fit_transform(
        df_analise[['Inverso_Custo', 'Inverso_Tempo_Medio', 'Potencial_PIB_Vizinhos']]
    )

    # 2.4 Modelo de Pontuação Ponderada
    pesos = {'Custo': 0.30, 'Logistica': 0.40, 'Consumo': 0.30}
    df_analise['Pontuacao_Final'] = (
        df_analise['Score_Custo'] * pesos['Custo'] +
        df_analise['Score_Logistica'] * pesos['Logistica'] +
        df_analise['Score_Consumo'] * pesos['Consumo']
    )

    # --- 3. Resultados e Visualização ---

    # 3.1 Definição da Cidade Vencedora
    cidade_vencedora = df_analise.loc[df_analise['Pontuacao_Final'].idxmax()]

    print("--- Análise de Localização Estratégica para Novo CD Magalu ---")
    print("\nDataframe de Análise Final:")
    print(df_analise[['Cidade', 'Custo_m2_Aluguel', 'Tempo_Medio_Entrega', 'Potencial_PIB_Vizinhos', 'Pontuacao_Final']].round(2))
    print("\n--- Conclusão ---")
    print(f"A cidade com a localização mais estratégica é: {cidade_vencedora['Cidade']}")
    print(f"Pontuação Final: {cidade_vencedora['Pontuacao_Final']:.2f}")

    # 3.2 Visualização Gráfica
    # Gráfico de Barras Comparativo
    fig_bar = px.bar(df_analise.sort_values('Pontuacao_Final', ascending=False),
                     x='Cidade', y='Pontuacao_Final',
                     title='Pontuação Final: Recife vs. Salvador',
                     color='Cidade', text=df_analise['Pontuacao_Final'].apply(lambda x: f'{x:.2f}'))
    fig_bar.show()

    # Mapa de Malha Viária (Exemplo com Plotly)
    # Coordenadas das cidades para visualização
    coords = {
        'Recife': {'lat': -8.0476, 'lon': -34.8770}, 'Salvador': {'lat': -12.9777, 'lon': -38.5016},
        'Aracaju': {'lat': -10.9167, 'lon': -37.05}, 'Maceió': {'lat': -9.6658, 'lon': -35.7353},
        'João Pessoa': {'lat': -7.1195, 'lon': -34.8451}, 'Natal': {'lat': -5.7833, 'lon': -35.2},
        'Fortaleza': {'lat': -3.7167, 'lon': -38.5167}, 'Teresina': {'lat': -5.0833, 'lon': -42.8},
        'São Luís': {'lat': -2.5333, 'lon': -44.3}
    }

    fig_map = go.Figure()

    # Adiciona as cidades principais
    fig_map.add_trace(go.Scattergeo(
        lon=[coords['Recife']['lon'], coords['Salvador']['lon']],
        lat=[coords['Recife']['lat'], coords['Salvador']['lat']],
        hoverinfo='text', text=['Recife (CD)', 'Salvador (CD)'],
        mode='markers', marker=dict(size=15, color='blue')))

    # Adiciona as capitais de destino
    for capital in capitais_ne:
        fig_map.add_trace(go.Scattergeo(
            lon=[coords[capital]['lon']], lat=[coords[capital]['lat']],
            hoverinfo='text', text=capital, mode='markers', marker=dict(size=8, color='gray')))

    # Adiciona as rotas a partir de Recife
    for capital in capitais_ne:
        fig_map.add_trace(go.Scattergeo(
            lon=[coords['Recife']['lon'], coords[capital]['lon']],
            lat=[coords['Recife']['lat'], coords[capital]['lat']],
            mode='lines', line=dict(width=2, color='green'),
            opacity=0.8, hoverinfo='none'))

    fig_map.update_layout(
        title_text='Malha Viária a partir de Recife (Verde)',
        showlegend=False,
        geo=dict(scope='south america',
                 projection_type='mercator',
                 center=dict(lat=-8, lon=-40),
                 lataxis_range=[-18, 0], lonaxis_range=[-50, -32])
    )
    fig_map.show()

if __name__ == '__main__':
    analisar_localizacao_cd()