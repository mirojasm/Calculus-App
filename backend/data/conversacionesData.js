const pool = require("../config/db");

const getConversaciones = async (usuario_id) => {
  const { rows } = await pool.query(
    `SELECT id, conversation_id, mensaje_usuario, respuesta_chatbot,
            current_exercise, creado_en
     FROM conversaciones
     WHERE usuario_id = $1
     ORDER BY creado_en DESC`,
    [usuario_id]
  );
  return rows;
};

module.exports = { getConversaciones };
