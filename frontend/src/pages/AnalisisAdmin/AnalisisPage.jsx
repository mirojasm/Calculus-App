import React, { useState, useEffect } from "react";
import { getAllUsers } from "../../services/registerService";
import { getAnalisis } from "../../services/analisisService";
import HeaderAdmin from "../../components/AdminHeader/Header";
import "bootstrap/dist/css/bootstrap.min.css";

const AnalisisPage = () => {
  const [users, setUsers] = useState([]);
  const [selectedUserId, setSelectedUserId] = useState("");
  const [analisis, setAnalisis] = useState(null);
  const [error, setError] = useState("");

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
    setAnalisis(null);
    setError("");
    if (!userId) return;
    try {
      const analisisData = await getAnalisis(userId);
      setAnalisis(analisisData);
    } catch (err) {
      setError("Error al cargar el análisis.");
    }
  };

  const parseAnalisisToHtml = (text) => {
    if (!text) return "";

    // Unir líneas de listas separadas tipo "1.\nTítulo\n: contenido" en una sola línea
    let html = text.replace(
      /^(\d+)\.\n([^\n]+)\n: ([^\n]+)/gm,
      "<li><strong>$2:</strong> $3</li>"
    );

    // Agrupar los <li> en <ol>
    html = html.replace(
      /((<li>.*?<\/li>\s*)+)/gs,
      (match) => `<ol>${match}</ol>`
    );

    // Títulos tipo ### Título o ## Título
    html = html.replace(
      /^#{2,3} (.*)$/gm,
      '<h4 class="mt-4 mb-2 fw-bold">$1</h4>'
    );
    // Negritas tipo **texto**
    html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    // Saltos de línea dobles a <br><br> (para separar párrafos)
    html = html.replace(/\n{2,}/g, "<br><br>");
    // Saltos de línea simples a <br>
    html = html.replace(/\n/g, "<br />");
    return html;
  };

  return (
    <>
      <HeaderAdmin />
      <div className="container py-5">
        <h2 className="mb-4">Análisis de Usuarios</h2>
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
            <h4 className="mb-3">Resultado del Análisis</h4>
            {analisis && analisis.analisis ? (
              <div
                className="card p-4 shadow-sm border-0 bg-light"
                dangerouslySetInnerHTML={{
                  __html: parseAnalisisToHtml(analisis.analisis),
                }}
              />
            ) : (
              <div className="alert alert-info">
                No hay análisis para este usuario.
              </div>
            )}
          </div>
        )}
      </div>
    </>
  );
};

export default AnalisisPage;
