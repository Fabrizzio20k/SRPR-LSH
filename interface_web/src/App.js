import React, { useState, useEffect } from 'react';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';
import './App.css';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);
const API_KEY = "ff95e997309cf4940136ed4cd79e568d"; 


const MovieCard = ({ movieId }) => {
  const [movieData, setMovieData] = useState({ title: `ID: ${movieId}`, posterUrl: '' });
  
  useEffect(() => {
    const fetchMovieData = async () => {
      const url = `https://api.themoviedb.org/3/movie/${movieId}?api_key=${API_KEY}&language=es-ES`;
      
      try {
        const response = await fetch(url);
        if (!response.ok) {
          setMovieData({ title: `ID: ${movieId}`, posterUrl: `notfound.png` });
          return;
        }
        const data = await response.json();
        
        const posterUrl = data.poster_path 
          ? `https://image.tmdb.org/t/p/w200${data.poster_path}`
          : `https://via.placeholder.com/150x225.png?text=${data.title.replace(/ /g, '+')}`;
        
        setMovieData({ title: data.title, posterUrl: posterUrl });
      } catch (error) {
        console.error("Error fetching movie data:", error);
        setMovieData({ title: `ID: ${movieId}`, posterUrl: `https://via.placeholder.com/150x225.png?text=Movie+${movieId}` });
      }
    };

    fetchMovieData();
  }, [movieId]);

  return (
    <div className="movie-card">
      <img src={movieData.posterUrl} alt={`Poster for ${movieData.title}`} />
      <p>{movieData.title}</p>
    </div>
  );
};

const RecommendationList = ({ title, recommendations, time, color }) => (
  <div className="recommendation-list">
    <h3 style={{ color }}>
      {title}
      <span className="time">{time ? `${time.toFixed(2)} ms` : ''}</span>
    </h3>
    <div className="movie-grid">
      {recommendations.map(rec => (
        <MovieCard key={rec.item_id} movieId={rec.item_id} />
      ))}
    </div>
  </div>
);
const QueryMetricsDisplay = ({ title, metrics, color }) => {
  console.log("Query Metrics:", metrics);
  if (!metrics) return null;
  return (
    <div className="recommendation-list2">
      <h3 style={{color: color}}>{title}</h3>
      <ul>
        <li><strong>Recall @K:</strong> {(metrics.recall * 100).toFixed(2)}%</li>
        <li><strong>nDCG @K:</strong> {metrics.ndcg.toFixed(4)}</li>
      </ul>
    </div>
  );
};


function App() {
  const [metrics, setMetrics] = useState([]);
  const [recommendations, setRecommendations] = useState(null);
  const [userId, setUserId] = useState('1');
  const [activeTab, setActiveTab] = useState('SRPR');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [kValue, setKValue] = useState(10); // <-- NUEVO estado para el valor de K
  const [queryMetrics, setQueryMetrics] = useState(null); // <-- NUEVO estado para las métricas

  useEffect(() => {
    fetch('/api/metrics')
      .then(res => res.json())
      .then(data => setMetrics(data))
      .catch(err => {
        console.error("Error fetching metrics:", err);
        setError("No se pudo conectar al backend C++. Asegúrate de que esté corriendo.");
      });
  }, []);

  const getRecommendations = () => {
    if (!userId) return;
    setLoading(true);
    setError('');
    setRecommendations(null);
    setQueryMetrics(null); // Limpiar métricas anteriores

    fetch(`/api/recommend?user_id=${userId}&k=${kValue}`)
      .then(res => res.json())
      .then(data => {
        if (data.error) {
          setError(`Error del backend: ${data.error}`);
        } else {
          console.log("Recommendations data:", data);
          setRecommendations(data);
          setQueryMetrics(data.query_metrics); // Guardar métricas de la consulta
        }
        setLoading(false);
      })
      .catch(err => {
        console.error("Error fetching recommendations:", err);
        setError("Error al conectar con el backend C++. Asegúrate de que esté corriendo.");
        setLoading(false);
      });
  };

  const chartData = {
    labels: ['nRecall@10', 'Recall@10', 'nDCG@10'],
    datasets: metrics.map((metric, index) => ({
      label: metric.model,
      data: [metric.n_recall, metric.recall, metric.ndcg],
      backgroundColor: metric.model.includes('SRPR') ? 'rgba(54, 162, 235, 0.6)' : 'rgba(255, 99, 132, 0.6)',
      borderColor: metric.model.includes('SRPR') ? 'rgba(54, 162, 235, 1)' : 'rgba(255, 99, 132, 1)',
      borderWidth: 1,
    })),
  };

  return (
    <div className="App">
      <header className="header">
        <h1>Dashboard de Resultados: SRPR vs. BPR</h1>
      </header>

      {error && <p style={{color: 'red', textAlign: 'center'}}>{error}</p>}
      
      <section className="metrics-container">
        <h2>Comparación de Métricas de Calidad con 1000 Usuarios</h2>
        <div className="chart-container">
          {metrics.length > 0 ? <Bar data={chartData} options={{ responsive: true }} /> : <p>Cargando métricas...</p>}
        </div>
      </section>
      
      <section className="recommendation-container">
        <h2>Buscador de Recomendaciones</h2>
        <div className="user-input-section">
          <input
            type="number"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            placeholder="Ingrese User ID"
          />
          <input
            type="number"
            value={kValue}
            onChange={(e) => setKValue(e.target.value)}
            placeholder="Top-K"
            style={{width: "80px"}}
          />
          <button onClick={getRecommendations} disabled={loading}>
            {loading ? 'Buscando...' : 'Obtener Recomendaciones'}
          </button>
        </div>
        
        {loading && <p style={{textAlign: 'center'}}>Obteniendo recomendaciones...</p>}

        {recommendations && (
          <div>
            <div className="tabs">
              <button className={`tab-button ${activeTab === 'SRPR' ? 'active' : ''}`} onClick={() => setActiveTab('SRPR')}>
                Modelo SRPR (Robusto)
              </button>
              <button className={`tab-button ${activeTab === 'BPR' ? 'active' : ''}`} onClick={() => setActiveTab('BPR')}>
                Modelo BPR (No Robusto)
              </button>
            </div>
            
            {activeTab === 'SRPR' && (
              <div className="results-grid">
                <RecommendationList title="Fuerza Bruta (Ground Truth SRPR)" recommendations={recommendations.srpr_ground_truth} time={recommendations.timings.srpr_brute_force_ms} color="#00695c" />
                <RecommendationList title="LSH con SRPR" recommendations={recommendations.srpr_lsh} time={recommendations.timings.srpr_lsh_ms} color="#2e7d32" />
                <QueryMetricsDisplay title="Métricas de esta Consulta (SRPR)" metrics={queryMetrics?.srpr} color="#00695c" />
              </div>
            )}
            
            {activeTab === 'BPR' && (
              <div className="results-grid">
                <RecommendationList title="Fuerza Bruta (Ground Truth BPR)" recommendations={recommendations.bpr_ground_truth} time={recommendations.timings.bpr_brute_force_ms} color="#c62828" />
                <RecommendationList title="LSH con BPR" recommendations={recommendations.bpr_lsh} time={recommendations.timings.bpr_lsh_ms} color="#d32f2f" />
                <QueryMetricsDisplay title="Métricas de esta Consulta (BPR)" metrics={queryMetrics?.bpr} color="#c62828" />
              </div>
            )}
          </div>
        )}
      </section>
    </div>
  );
}

export default App;