import React, { useState, useEffect } from "react";
import { getAllUsers } from "../../services/registerService";
import { getResultados } from "../../services/resultadoService";
import HeaderAdmin from "../../components/AdminHeader/Header";
import "katex/dist/katex.min.css";
import { InlineMath } from "react-katex";
import "bootstrap/dist/css/bootstrap.min.css";
import "./ResultadosPage.css";

const ResultadosPage = () => {
  const [users, setUsers] = useState([]);
  const [selectedUserId, setSelectedUserId] = useState("");
  const [resultados, setResultados] = useState([]);
  const [error, setError] = useState("");
  
  const extractLatex = (input) => {
  const match = input.match(/\\\((.*?)\\\)/);
  return match ? match[1] : input;
};

  useEffect(() => {
    const fetchUsers = async () => {
      try {
        const usersData = await getAllUsers();
        setUsers(usersData);
      } catch (err) {
        setError("Error al cargar los usuarios.");
      }
    };
    fetchUsers();
  }, []);

  const handleUserSelect = async (e) => {
    const userId = parseInt(e.target.value, 10);
    setSelectedUserId(userId);
    setResultados([]);
    setError("");
    if (!userId) return;
    try {
      const resultadosData = await getResultados(userId);
      setResultados(resultadosData);
    } catch (err) {
      setError("Error al cargar los resultados.");
    }
  };

  return (
    <>
      <HeaderAdmin />
      <div className="container py-5">
        <h2 className="mb-4">Resultados de Usuarios</h2>
        <div className="mb-4">
          <label htmlFor="userSelect" className="form-label fw-bold">
            Selecciona un usuario:
          </label>
          <select
            className="form-select"
            id="userSelect"
            value={selectedUserId}
            onChange={handleUserSelect}
          >
            <option value="">-- Elige un usuario --</option>
            {users.map((user) => (
              <option key={user.id} value={user.id}>
                {user.nombre}
              </option>
            ))}
          </select>
        </div>
        {selectedUserId && (
          <div>
            <h4 className="mb-3">Resultados</h4>
            {resultados.length > 0 ? (
              <pre className="plain-text-conversation">
               <div className="row">
  {resultados.map((res, index) => (
    <div key={res.id} className="col-md-6 mb-4">
      <div className={`card h-100 shadow-sm border-${res.estado_ejercicio === "bueno" ? "success" : "danger"}`}>
        <div className="card-body">
          <h5 className="card-title">Ejercicio {res.current_exercise}</h5>
          <p className="card-text"><strong>Respuesta del alumno:</strong></p>
          <div className="alert alert-secondary">
            <InlineMath math={extractLatex(res.resultado_alumno)} />
          </div>
          <p className="card-text"><strong>Respuesta correcta:</strong></p>
          <div className="alert alert-light">
            <InlineMath math={extractLatex(res.resultado_correcto)} />
          </div>
          <span className={`badge bg-${res.estado_ejercicio === "bueno" ? "success" : "danger"}`}>
            {res.estado_ejercicio === "bueno" ? "Correcto" : "Incorrecto"}
          </span>
          <p className="text-muted mt-2 mb-0" style={{ fontSize: "0.9rem" }}>
            {new Date(res.creado_en).toLocaleString("es-CL")}
          </p>
        </div>
      </div>
    </div>
  ))}
</div>
              </pre>
            ) : (
              <div className="alert alert-info">No hay resultados para este usuario.</div>
            )}
          </div>
        )}
        {error && <div className="alert alert-danger mt-3">{error}</div>}
      </div>
    </>
  );
};

export default ResultadosPage;
