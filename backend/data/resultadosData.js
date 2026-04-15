const pool = require("../config/db");

const createResultado = async (
  usuario_id,
  resultado_alumno,
  resultado_correcto,
  estado_ejercicio,
  current_exercise,
  timestamp
) => {
  const result = await pool.query(
    "INSERT INTO resultados(usuario_id,resultado_alumno,resultado_correcto,estado_ejercicio,current_exercise,creado_en) VALUES ($1, $2, $3, $4, $5, $6) RETURNING *",
    [
      usuario_id,
      resultado_alumno,
      resultado_correcto,
      estado_ejercicio,
      current_exercise,
      timestamp,
    ]
  );
  return result.rows[0];
};

const getResultadosPorUsuario = async (usuario_id) => {
  const result = await pool.query(
    "SELECT * FROM resultados WHERE usuario_id = $1 ORDER BY creado_en DESC",
    [usuario_id]
  );
  return result.rows;
};

module.exports = { createResultado, getResultadosPorUsuario };
