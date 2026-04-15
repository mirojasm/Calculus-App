const pool = require("../config/db");

const createAccion= async (usuario_id,accion,current_exercise,timestamp) => {
    const result = await pool.query(
      "INSERT INTO acciones(usuario_id,accion,current_exercise,creado_en) VALUES ($1, $2, $3, $4) RETURNING *",
      [usuario_id,accion,current_exercise,timestamp]
    );
    return result.rows[0];
  };

module.exports = {createAccion};