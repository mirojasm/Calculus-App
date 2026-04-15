const pool = require("../config/db");

const createAnalisis= async (usuario_id, analisis,puntaje, creado_en) => {
    const result = await pool.query(
      "INSERT INTO analisis (usuario_id,analisis,puntaje,creado_en) VALUES ($1, $2, $3, $4) RETURNING *",
      [usuario_id, analisis,puntaje, creado_en]
    );
    return result.rows[0];
  };
  
const getAnalisis= async (usuario_id) => {
    const result = await pool.query(
        "SELECT * FROM analisis WHERE usuario_id = $1",
        [usuario_id]
      );
    
    return result.rows[0];
  };

  module.exports = {
    createAnalisis,
    getAnalisis
  };




