const pool = require("../config/db");

const createChat = async (
  usuario_id,
  conversation_id,
  mensaje_usuario,
  respuesta_chatbot,
  current_exercise,
  creado_en
) => {
  const { rows } = await pool.query(
    `INSERT INTO conversaciones
       (usuario_id, conversation_id, mensaje_usuario, respuesta_chatbot, current_exercise, creado_en)
     VALUES ($1, $2, $3, $4, $5, $6)
     RETURNING *`,
    [usuario_id, conversation_id, mensaje_usuario, respuesta_chatbot, current_exercise, creado_en]
  );
  return rows[0];
};

const getConversationTurns = async (usuario_id, current_exercise) => {
  const { rows } = await pool.query(
    `SELECT mensaje_usuario, respuesta_chatbot, creado_en
     FROM conversaciones
     WHERE usuario_id = $1 AND current_exercise = $2
     ORDER BY creado_en ASC`,
    [usuario_id, current_exercise]
  );
  return rows;
};

const countUserMessages = async (usuario_id, current_exercise) => {
  const { rows } = await pool.query(
    `SELECT COUNT(*)::int AS count
     FROM conversaciones
     WHERE usuario_id = $1 AND current_exercise = $2`,
    [usuario_id, current_exercise]
  );
  return rows[0].count;
};

module.exports = { createChat, getConversationTurns, countUserMessages };
