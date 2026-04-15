import React, { useState, useEffect } from "react";
import { getAllUsers } from "../../services/registerService";
import { getUserConversations } from "../../services/conversationService";
import { format } from "date-fns";
import HeaderAdmin from "../../components/AdminHeader/Header";
import "bootstrap/dist/css/bootstrap.min.css";
import "./ConversationsPage.css";

const ConversationsPage = () => {
  const [users, setUsers] = useState([]);
  const [selectedUserId, setSelectedUserId] = useState("");
  const [conversations, setConversations] = useState([]);
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
    setConversations([]);
    setError("");

    if (!userId) return;

    try {
      const conversationsData = await getUserConversations(userId);
      setConversations(conversationsData);
    } catch (err) {
      setError("Error al cargar las conversaciones.");
    }
  };

return (
  <>
    <HeaderAdmin />
    <div className="container py-5">
      <h2 className="mb-4">Conversaciones de Usuarios</h2>

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
          <h4 className="mb-3">Historial de Conversaciones</h4>
          {conversations.length > 0 ? (
            <pre className="plain-text-conversation">
  {conversations.map((conv, index) => (
    <div key={index}>
      <strong>🧑 Usuario:</strong> {conv.mensaje_usuario}
      <br />
      <strong>🤖 Chatbot:</strong> {conv.respuesta_chatbot}
      <br />
      <span className="timestamp">
        🕒 {format(new Date(conv.creado_en), "dd/MM/yyyy HH:mm:ss")} | 🧩 {conv.currentexercise}
      </span>
      <br />
      <br />
    </div>
  ))}
</pre>
          ) : (
            <div className="alert alert-info">No hay conversaciones para este usuario.</div>
          )}
        </div>
      )}

      {error && <div className="alert alert-danger mt-3">{error}</div>}
    </div>
  </>
);
};

export default ConversationsPage;


